from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import subprocess
import os
import uuid
import glob

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_DIR = 'uploads/'
TEMP_DIR = 'temp_clips/'
OUTPUT_DIR = 'outputs/'
TEMPLATES_DIR = 'templates/'
DEBUG_DIR = 'debug_frames/'

# Valorant Killfeed Turkuaz / Nane Yeşili HSV Aralığı
LOWER_CYAN = np.array([65, 50, 100])
UPPER_CYAN = np.array([105, 255, 255])

# CS2 Killfeed Kırmızı / Turuncu Bildirim HSV Aralıkları
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])
# ----------------------------------------

for d in [UPLOAD_DIR, TEMP_DIR, OUTPUT_DIR, DEBUG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def load_templates(directory):
    templates = []
    for filepath in glob.glob(os.path.join(directory, '*.png')):
        filename = os.path.basename(filepath)
        if 'headshot' in filename.lower():
            continue
        tpl = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if tpl is not None:
            templates.append((filename, tpl))
    print(f"Toplam {len(templates)} adet silah şablonu yüklendi.")
    return templates

def detect_valorant_kills(video_path, templates):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
        
    kill_timestamps = []
    frame_count = 0
    skip_frames = max(1, int(fps / 4))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        height, width = frame.shape[:2]
        
        roi_x1 = int(width * 0.72)
        roi_y1 = int(height * 0.04)
        roi_x2 = width
        roi_y2 = int(height * 0.28)
        
        roi_bgr = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(roi_hsv, LOWER_CYAN, UPPER_CYAN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if (130 < w < 280) and (22 < h < 42):
                timestamp = frame_count / fps
                kill_timestamps.append(timestamp)
                
                print(f"[!] Valorant Kill Yakalandı: {timestamp:.2f}. sn (Boyut: {w}x{h})")
                
                debug_frame = roi_bgr.copy()
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"val_feed_{timestamp:.2f}.png"), debug_frame)
                
                frame_count += int(fps * 3.0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                break

    cap.release()
    return kill_timestamps

def detect_cs2_kills(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    kill_timestamps = []
    frame_count = 0
    skip_frames = max(1, int(fps / 5))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        height, width = frame.shape[:2]

        roi_x1 = int(width * 0.70)
        roi_y1 = int(height * 0.02)
        roi_x2 = int(width * 0.98)
        roi_y2 = int(height * 0.25)

        roi_bgr = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(roi_hsv, LOWER_RED1, UPPER_RED1)
        mask2 = cv2.inRange(roi_hsv, LOWER_RED2, UPPER_RED2)
        mask = mask1 | mask2

        red_pixels = cv2.countNonZero(mask)

        if red_pixels > 450:
            timestamp = frame_count / fps
            kill_timestamps.append(timestamp)

            print(f"[!] CS2 Kill Yakalandı: {timestamp:.2f}. sn (Kırmızı Piksel: {red_pixels})")

            debug_frame = roi_bgr.copy()
            cv2.imwrite(os.path.join(DEBUG_DIR, f"cs2_feed_{timestamp:.2f}.png"), debug_frame)

            frame_count += int(fps * 2.5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()
    return kill_timestamps

def merge_segments(timestamps, pre_roll=3.5, post_roll=2.5):
    if not timestamps:
        return []
        
    segments = []
    for ts in timestamps:
        start = max(0.0, ts - pre_roll)
        end = ts + post_roll
        
        if not segments:
            segments.append([start, end])
        else:
            last_segment = segments[-1]
            if start <= last_segment[1]:
                last_segment[1] = max(last_segment[1], end)
            else:
                segments.append([start, end])
    return segments

def process_video(input_video, output_video, format_type='9-16', game='valorant'):
    job_id = str(uuid.uuid4())[:8]
    print(f"[{job_id}] İşlem başladı (Oyun: {game}, Format: {format_type})...")
    
    if game.lower() == 'cs2':
        timestamps = detect_cs2_kills(input_video)
    else:
        templates = load_templates(TEMPLATES_DIR)
        timestamps = detect_valorant_kills(input_video, templates)
    
    if not timestamps:
        print(f"[{job_id}] Hiç geçerli kill bulunamadı.")
        return False
        
    segments = merge_segments(timestamps)
    print(f"[{job_id}] Kırpılacak Kesitler: {segments}")
    
    # 9:16 modunda oyun bazlı killfeed kırpma alanı
    if game.lower() == 'cs2':
        vertical_filter = (
            "[0:v]scale=-1:1920,crop=1080:1920:(iw-1080)/2:0[main];"
            "[0:v]crop=w=iw*0.30:h=ih*0.22:x=iw*0.69:y=ih*0.02,scale=920:-1[kf];"
            "[main][kf]overlay=(W-w)/2:120[outv]"
        )
    else:
        vertical_filter = (
            "[0:v]scale=-1:1920,crop=1080:1920:(iw-1080)/2:0[main];"
            "[0:v]crop=w=iw*0.32:h=ih*0.20:x=iw*0.68:y=ih*0.02,scale=900:-1[kf];"
            "[main][kf]overlay=(W-w)/2:140[outv]"
        )
    
    temp_files = []
    concat_list_path = os.path.join(TEMP_DIR, f"concat_{job_id}.txt")
    
    with open(concat_list_path, 'w', encoding='utf-8') as f:
        for idx, (start, end) in enumerate(segments):
            duration = end - start
            temp_output = os.path.join(TEMP_DIR, f"clip_{job_id}_{idx}.mp4")
            
            if format_type == '16-9':
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-ss', f"{start:.2f}", '-t', f"{duration:.2f}",
                    '-i', input_video,
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-c:a', 'aac',
                    temp_output
                ]
            else:
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-ss', f"{start:.2f}", '-t', f"{duration:.2f}",
                    '-i', input_video,
                    '-filter_complex', vertical_filter,
                    '-map', '[outv]', '-map', '0:a',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                    '-c:a', 'aac',
                    temp_output
                ]
            
            print(f"[{job_id}] Kesit hazırlanıyor: {idx+1}/{len(segments)} ({start:.2f}s - {end:.2f}s)")
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            f.write(f"file '{os.path.abspath(temp_output)}'\n")
            temp_files.append(temp_output)
            
    print(f"[{job_id}] Klipler birleştiriliyor...")
    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
        output_video
    ]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    for tmp in temp_files:
        if os.path.exists(tmp):
            os.remove(tmp)
            
    print(f"[{job_id}] BİTTİ! Dosya hazır: {output_video}")
    return True

def detect_cs2_kills(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    detected_kills = []
    frame_step = max(1, int(fps * 0.15))
    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        frame_1080 = cv2.resize(frame, (1920, 1080))
        # CS2 sağ üst killfeed kutusu
        roi = frame_1080[20:250, 1350:1900]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = cv2.countNonZero(mask)

        if red_pixels > 450:
            timestamp = current_frame / fps
            if len(detected_kills) == 0 or (timestamp - detected_kills[-1] > 2.5):
                detected_kills.append(timestamp)
                print(f"[!] CS2 Kill Yakalandı: {timestamp:.2f}. sn (Piksel: {red_pixels})", flush=True)
                current_frame += int(fps * 2.0)
                continue

        current_frame += frame_step

    cap.release()
    return detected_kills

# ---------------- API ENDPOINT ----------------
@app.route('/api/process', methods=['POST'])
def handle_process():
    if 'video' not in request.files:
        return jsonify({'error': 'Video dosyası bulunamadı'}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    selected_format = request.form.get('format', '9-16')
    selected_game = request.form.get('game', 'valorant')  # 'cs2' veya 'valorant'
    
    uid = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"input_{uid}_{file.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"montage_{uid}.mp4")
    
    file.save(input_path)
    
    success = process_video(input_path, output_path, format_type=selected_format, game=selected_game)
    
    if os.path.exists(input_path):
        os.remove(input_path)
        
    if not success:
        return jsonify({'error': f'{selected_game.upper()} videosunda kill bulunamadı'}), 400
        
    return send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f"{selected_game}_{selected_format}_{uid}.mp4"
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)