import cv2
import numpy as np
import subprocess
import os
import uuid
import glob

# ---------------- CONFIG ----------------
TEMPLATES_DIR = 'templates/'
TEMP_DIR = 'temp_clips/'
DEBUG_DIR = 'debug_frames/'

# Valorant Killfeed Turkuaz / Nane Yeşili HSV Aralığı
LOWER_CYAN = np.array([65, 50, 100])
UPPER_CYAN = np.array([105, 255, 255])
# ----------------------------------------

for d in [TEMP_DIR, DEBUG_DIR]:
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

def detect_kills(video_path, templates):
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
                
                print(f"[!] GERÇEK Killfeed Yakalandı: {timestamp:.2f}. sn (Boyut: {w}x{h})")
                
                debug_frame = roi_bgr.copy()
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"real_kill_{timestamp:.2f}.png"), debug_frame)
                
                frame_count += int(fps * 3.0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                break

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

def process_video(input_video, output_video):
    job_id = str(uuid.uuid4())[:8]
    print(f"[{job_id}] İşlem başladı...")
    
    templates = load_templates(TEMPLATES_DIR)
    timestamps = detect_kills(input_video, templates)
    
    if not timestamps:
        print(f"[{job_id}] Hiç geçerli kill bulunamadı.")
        return False
        
    segments = merge_segments(timestamps)
    print(f"[{job_id}] Kırpılacak Kesitler: {segments}")
    
    # --- 3 KATMANLI DİKEY SHORT FORMATI FİLTRESİ ---
  # 1. [main]: 16:9 videoyu 1920 yüksekliğe ölçekleyip tam ortadan 1080x1920 doğal dikey kırpar (Crosshair doğal merkezde)
    # 2. [kf]: Sağ üstteki killfeed alanını kesip 900px genişliğe büyütür
    # 3. Killfeed'i üst kısma (Y: 140) başlık gibi ortalayarak yerleştirir
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

if __name__ == "__main__":
    test_video = "test_gameplay.mp4"
    output_video = "final_shorts.mp4"
    if os.path.exists(test_video):
        process_video(test_video, output_video)
    else:
        print(f"Hata: {test_video} bulunamadı!")