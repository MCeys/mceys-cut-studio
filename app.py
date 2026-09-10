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
    prev_region = None
    
    # 0.2 saniyede bir tara (daha hassas)
    frame_step = max(1, int(fps * 0.20))
    current_frame = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (640, 360))

        # Sağ üst Valorant Killfeed alanı
        kf_y1, kf_y2 = int(360 * 0.01), int(360 * 0.16)
        kf_x1, kf_x2 = int(640 * 0.72), int(640 * 0.99)
        cur_region = resized[kf_y1:kf_y2, kf_x1:kf_x2]

        if prev_region is not None:
            diff = cv2.absdiff(cur_region, prev_region)
            avg_delta = np.mean(diff)

            b = cur_region[:, :, 0].astype(int)
            g = cur_region[:, :, 1].astype(int)
            r = cur_region[:, :, 2].astype(int)

            # Eşik değerleri daha hassas hale getirildi
            red_mask = (r > 125) & (r > g + 25) & (r > b + 20)
            team_mask = (g > 105) & (b > 95) & (g > r + 15)

            red_count = np.count_nonzero(red_mask)
            team_count = np.count_nonzero(team_mask)

            is_kill = False
            if mode == 'solo':
                if avg_delta > 15 and red_count >= 12:
                    is_kill = True
            else:
                if avg_delta > 15 and (red_count >= 12 or team_count >= 12):
                    is_kill = True

            timestamp = current_frame / fps
            if is_kill:
                if len(detected_kills) == 0 or (timestamp - detected_kills[-1] > 2.0):
                    detected_kills.append(timestamp)

        prev_region = cur_region
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

    # 1. OpenCV ile kill anlarını yakala
    kill_times = detect_kills_opencv(input_path, mode=kill_mode)
    print(f"[{unique_id}] Yakalanan Kill Zamanlari:", kill_times, flush=True)

    # 2. Kill anlarının 3.5 sn öncesi ve 1.5 sn sonrasını topla
    segments = []
    for kt in kill_times:
        start_t = max(0.0, float(kt) - 3.5)
        end_t = float(kt) + 1.5
        if segments and start_t <= segments[-1][1]:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end_t))
        else:
            segments.append((start_t, end_t))

    # Eğer video boyunca tek bir kill bile okuyamadıysa en azından ilk 10 saniyeyi al
    if not segments:
        segments = [(0.0, 10.0)]

    filter_complex_parts = []
    concat_inputs = []

    for idx, (st, et) in enumerate(segments):
        v_trim = f"[0:v]trim=start={st}:end={et},setpts=PTS-STARTPTS"
        a_trim = f"[0:a]atrim=start={st}:end={et},asetpts=PTS-STARTPTS[a{idx}]"

        if target_format == '9-16':
            # Valorant Profesyonel Shorts Şablonu:
            # 1. Arka plan: 1080x1920 blur
            # 2. Orta katman: Oyunun ana aksiyon/crosshair alanı (büyütülmüş ve ortalanmış)
            # 3. Üst katman: Sağ üstteki Killfeed'in büyütülüp tepeye yapıştırılmış hali
            filter_complex_parts.append(
                f"{v_trim},split=3[bg_raw{idx}][fg_raw{idx}][kf_raw{idx}];"
                f"[bg_raw{idx}]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg{idx}];"
                f"[fg_raw{idx}]scale=1080:-1[fg{idx}];"
                f"[bg{idx}][fg{idx}]overlay=(W-w)/2:(H-h)/2[base{idx}];"
                f"[kf_raw{idx}]crop=iw*0.25:ih*0.20:iw*0.75:0,scale=540:-1[kf_zoom{idx}];"
                f"[base{idx}][kf_zoom{idx}]overlay=(W-w)/2:120[v{idx}];"
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
        return send_file(output_path, as_attachment=True, download_name=f"valorant_highlight_{unique_id}.mp4")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "FFmpeg montajlama hatasi", "details": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)