import os
import glob
import subprocess
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/var/www/cut-backend/storage'
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Başlangıçta tüm şablonları gri tonlamalı ve kenar algılamalı belleğe al
LOADED_TEMPLATES = []

def load_templates():
    global LOADED_TEMPLATES
    LOADED_TEMPLATES = []
    template_files = glob.glob(os.path.join(TEMPLATE_DIR, '*.png'))
    for tf in template_files:
        tpl = cv2.imread(tf, cv2.IMREAD_UNCHANGED)
        if tpl is not None:
            # Şeffaf PNG ise alfa kanalını maske olarak ayır veya griye çevir
            if tpl.shape[-1] == 4:
                gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2GRAY)
            else:
                gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            
            # Kenar algılama (Canny) ile arka plan rengini tamamen devre dışı bırakıyoruz
            edges = cv2.Canny(gray_tpl, 50, 150)
            LOADED_TEMPLATES.append({
                "name": os.path.basename(tf),
                "gray": gray_tpl,
                "edges": edges,
                "shape": gray_tpl.shape[:2]
            })
    print(f"Toplam {len(LOADED_TEMPLATES)} adet silah şablonu yüklendi.", flush=True)

load_templates()

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "templates_loaded": len(LOADED_TEMPLATES),
        "message": "Template Matching motoru hazır!"
    }), 200

def detect_kills_template(video_path):
    if not LOADED_TEMPLATES:
        print("UYARI: templates/ klasöründe şablon görseli bulunamadı!", flush=True)
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detected_kills = []
    # 0.15 saniyede bir tara
    frame_step = max(1, int(fps * 0.15))
    current_frame = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        frame_1080 = cv2.resize(frame, (1920, 1080))
        # Tam çizdiğin o 1080p siyah kutu
        strict_kf = frame_1080[25:150, 1520:1905]

        kf_gray = cv2.cvtColor(strict_kf, cv2.COLOR_BGR2GRAY)
        kf_edges = cv2.Canny(kf_gray, 50, 150)

        match_found = False

        # Kutuda tüm silah silüetlerini ara
        for tpl in LOADED_TEMPLATES:
            th, tw = tpl["shape"]
            if kf_edges.shape[0] < th or kf_edges.shape[1] < tw:
                continue

            # Şablon kenar eşleştirmesi: Renk farkından etkilenmez
            res = cv2.matchTemplate(kf_edges, tpl["edges"], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            # Silah silüeti %72 ve üstü eşleştiği an killfeed düşmüştür!
            if max_val >= 0.72:
                match_found = True
                break

        timestamp = current_frame / fps

        if match_found:
            if len(detected_kills) == 0 or (timestamp - detected_kills[-1] > 2.5):
                detected_kills.append(timestamp)

        current_frame += frame_step

    cap.release()
    return detected_kills

@app.route('/api/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "Video dosyasi bulunamadi"}), 400

    video_file = request.files['video']
    target_format = request.form.get('format', '9-16')

    unique_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"input_{unique_id}.mp4")
    output_path = os.path.join(UPLOAD_FOLDER, f"output_{unique_id}.mp4")

    video_file.save(input_path)

    kill_times = detect_kills_template(input_path)
    print(f"[{unique_id}] Şablon Eşleşmesiyle Yakalanan Kill Zamanlari:", kill_times, flush=True)

    segments = []
    for kt in kill_times:
        start_t = max(0.0, float(kt) - 2.5)
        end_t = float(kt) + 1.0
        if segments and start_t <= segments[-1][1]:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end_t))
        else:
            segments.append((start_t, end_t))

    if not segments:
        segments = [(2.0, 8.0)]

    filter_complex_parts = []
    concat_inputs = []

    for idx, (st, et) in enumerate(segments):
        v_trim = f"[0:v]trim=start={st}:end={et},setpts=PTS-STARTPTS"
        a_trim = f"[0:a]atrim=start={st}:end={et},asetpts=PTS-STARTPTS[a{idx}]"

        if target_format == '9-16':
            filter_complex_parts.append(
                f"{v_trim},split=2[main_raw{idx}][kf_raw{idx}];"
                f"[main_raw{idx}]crop=607:1080:656:0,scale=1080:1920[main{idx}];"
                f"[kf_raw{idx}]crop=385:125:1520:25,scale=760:-1[kf_zoom{idx}];"
                f"[main{idx}][kf_zoom{idx}]overlay=(W-w)/2:120[v{idx}];"
                f"{a_trim}"
            )
        else:
            filter_complex_parts.append(f"{v_trim}[v{idx}];{a_trim}")

        concat_inputs.append(f"[v{idx}][a{idx}]")

    n_seg = len(segments)
    concat_str = "".join(concat_inputs) + f"concat=n={n_seg}:v=1:a=1[vout][aout]"
    full_filter = ";".join(filter_complex_parts) + ";" + concat_str

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter_complex', full_filter,
        '-map', '[vout]', '-map', '[aout]',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22',
        '-c:a', 'aac',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        return send_file(output_path, as_attachment=True, download_name=f"valorant_shorts_{unique_id}.mp4")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "FFmpeg montajlama hatasi", "details": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)