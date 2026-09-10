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

def detect_kills_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detected_kills = []
    frame_step = max(1, int(fps * 0.20))
    current_frame = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Videoyu 1920x1080 standartına oturt
        frame_1080 = cv2.resize(frame, (1920, 1080))

        # ÇİZDİĞİN SİYAH KUTUNUN TAM 1920x1080 PİKSEL KOORDİNATLARI:
        # Y: 25 ile 150 arası | X: 1520 ile 1905 arası
        strict_kf = frame_1080[25:150, 1520:1905]

        hsv = cv2.cvtColor(strict_kf, cv2.COLOR_BGR2HSV)

        # 1. Valorant Kırmızı Düşman Kutusu
        r1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
        red_mask = r1 | r2
        red_count = cv2.countNonZero(red_mask)

        # 2. Killfeed İçi Beyaz Silah/Yetenek İkonu
        white_mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 45, 255]))
        white_count = cv2.countNonZero(white_mask)

        # 1920x1080 kutusunda bir kill düşerse:
        # Kırmızı piksel alanı en az 350 piksel parlar, beyaz silah ikonu en az 60 piksel olur.
        if (350 <= red_count <= 4500) and white_count >= 60:
            timestamp = current_frame / fps
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

    kill_times = detect_kills_opencv(input_path)
    print(f"[{unique_id}] 1080p Kutudan Yakalanan Kill Zamanlari:", kill_times, flush=True)

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
            # 1. Ana Aksiyon: 1920x1080 videonun ortasındaki 607x1080 alanı alıp 1080x1920'ye büyütür (tam ekran Shorts)
            # 2. Killfeed: O siyah kutuyu (1520,25) kesip 760 genişliğe ölçekler ve üste yapıştırır
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