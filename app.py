import os
import subprocess
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/var/www/cut-backend/storage'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Python OpenCV motoru devrede!"}), 200

def detect_kills_opencv(video_path, mode='team'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detected_kills = []
    
    # 0.2 saniyede bir tara
    frame_step = max(1, int(fps * 0.20))
    current_frame = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # 1280x720 baz alarak sağ üst killfeed şeridine kilitlen
        resized = cv2.resize(frame, (1280, 720))
        # y: 20-140, x: 1000-1260 arası (Doğrudan killfeed şeritleri)
        kf_roi = resized[20:140, 1000:1260]

        # BGR'dan HSV'ye dönüştür
        hsv = cv2.cvtColor(kf_roi, cv2.COLOR_BGR2HSV)

        # 1. VALORANT KIRMIZISI (Sağdaki ölen düşman kutusu)
        red_mask1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2

        # 2. TURKUAZ / CAMGÖBEĞİ (Soldaki takım/bizim kutu)
        cyan_mask = cv2.inRange(hsv, np.array([75, 80, 100]), np.array([95, 255, 255]))

        red_pixels = cv2.countNonZero(red_mask)
        cyan_pixels = cv2.countNonZero(cyan_mask)

        is_kill = False
        if mode == 'solo':
            # Sadece bizim vuruşlarımız (kırmızı hedef + turkuaz katil kutusu)
            if red_pixels > 80 and cyan_pixels > 80:
                is_kill = True
        else:
            # Tüm takım kill'leri (kırmızı düşman kutusu düştüğü an)
            if red_pixels > 80:
                is_kill = True

        timestamp = current_frame / fps
        if is_kill:
            # Aynı kill bandı ekranda kaldığı için mükerrer saymayı engelle (2.5 sn aralık)
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
    kill_mode = request.form.get('mode', 'team')

    unique_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"input_{unique_id}.mp4")
    output_path = os.path.join(UPLOAD_FOLDER, f"output_{unique_id}.mp4")

    video_file.save(input_path)

    kill_times = detect_kills_opencv(input_path, mode=kill_mode)
    print(f"[{unique_id}] Yakalanan Gercek Kill Zamanlari:", kill_times, flush=True)

    # Her kill'in 2.5 sn öncesi ve 1.0 sn sonrasını topla
    segments = []
    for kt in kill_times:
        start_t = max(0.0, float(kt) - 2.5)
        end_t = float(kt) + 1.0
        if segments and start_t <= segments[-1][1]:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end_t))
        else:
            segments.append((start_t, end_t))

    # Eğer hiç kill bulamazsa videonun ortasından 8 saniyelik bir kesit al
    if not segments:
        segments = [(2.0, 10.0)]

    filter_complex_parts = []
    concat_inputs = []

    for idx, (st, et) in enumerate(segments):
        v_trim = f"[0:v]trim=start={st}:end={et},setpts=PTS-STARTPTS"
        a_trim = f"[0:a]atrim=start={st}:end={et},asetpts=PTS-STARTPTS[a{idx}]"

        if target_format == '9-16':
            # GERÇEK 9:16 SHORTS DÜZENİ:
            # 1. Ana Ekran: Tam ortaya crosshair ve aksiyon alanı crop edilir (1080x1920 tam ekran)
            # 2. Üst Bar: Sağ üstteki killfeed şeridi büyütülüp tepeye ortalanır
            filter_complex_parts.append(
                f"{v_trim},split=2[main_raw{idx}][kf_raw{idx}];"
                f"[main_raw{idx}]crop=ih*9/16:ih:(iw-ow)/2:0,scale=1080:1920[main{idx}];"
                f"[kf_raw{idx}]crop=iw*0.25:ih*0.16:iw*0.75:0,scale=720:-1[kf_zoom{idx}];"
                f"[main{idx}][kf_zoom{idx}]overlay=(W-w)/2:100[v{idx}];"
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