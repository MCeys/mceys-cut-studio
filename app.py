import os
import json
import subprocess
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/var/www/cut-backend/storage'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Python FFmpeg motoru devrede!"}), 200

@app.route('/api/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "Video dosyasi bulunamadi"}), 400

    video_file = request.files['video']
    target_format = request.form.get('format', '9-16')
    raw_kills = request.form.get('kills', '[]')

    try:
        kill_times = json.loads(raw_kills)
    except Exception:
        kill_times = []

    unique_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"input_{unique_id}.mp4")
    output_path = os.path.join(UPLOAD_FOLDER, f"output_{unique_id}.mp4")

    video_file.save(input_path)

    # Vurus anlarinin 3.5 sn oncesi ve 1.5 sn sonrasini kes
    segments = []
    for kt in kill_times:
        start_t = max(0.0, float(kt) - 3.5)
        end_t = float(kt) + 1.5
        if segments and start_t <= segments[-1][1]:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end_t))
        else:
            segments.append((start_t, end_t))

    if not segments:
        segments = [(0.0, 15.0)]

    filter_complex_parts = []
    concat_inputs = []

    for idx, (st, et) in enumerate(segments):
        v_trim = f"[0:v]trim=start={st}:end={et},setpts=PTS-STARTPTS"
        a_trim = f"[0:a]atrim=start={st}:end={et},asetpts=PTS-STARTPTS[a{idx}]"

        if target_format == '9-16':
            filter_complex_parts.append(
                f"{v_trim},split=2[bg_raw{idx}][fg_raw{idx}];"
                f"[bg_raw{idx}]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg{idx}];"
                f"[fg_raw{idx}]scale=1080:-1[fg{idx}];"
                f"[bg{idx}][fg{idx}]overlay=(W-w)/2:(H-h)/2[v{idx}];"
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