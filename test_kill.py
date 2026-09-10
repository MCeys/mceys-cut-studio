import cv2, glob, os, numpy as np

files = sorted(glob.glob('/var/www/cut-backend/storage/input_*.mp4'), key=os.path.getmtime)
if not files:
    print('Depoda video yok!')
    exit()

cap = cv2.VideoCapture(files[-1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

step = int(fps * 0.5)
for f in range(0, total_frames, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret: break
    resized = cv2.resize(frame, (1280, 720))
    kf_roi = resized[20:140, 1000:1260]
    hsv = cv2.cvtColor(kf_roi, cv2.COLOR_BGR2HSV)
    
    r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_count = cv2.countNonZero(r1 | r2)
    
    sec = f / fps
    if red_count > 10:
        print(f'{sec:.1f}s -> Kirmizi: {red_count}')
cap.release()