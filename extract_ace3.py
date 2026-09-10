import os
import cv2
import numpy as np

img = cv2.imread('killfeed-ace_3.jpg')
if img is None:
    img = cv2.imread('killfeed-ace_3.png')

if img is None:
    print("HATA: 'killfeed-ace_3.jpg' veya .png dosyasi bulunamadi!")
    exit()

os.makedirs('templates', exist_ok=True)
h, w, _ = img.shape

# 5 sıranın dikey aralıkları
rows = [
    ("outlaw",      int(h * 0.00), int(h * 0.20), False),
    ("ares",        int(h * 0.20), int(h * 0.40), False),
    ("odin",        int(h * 0.40), int(h * 0.60), False),
    ("golden_gun",  int(h * 0.60), int(h * 0.80), True),  # Sari tabanca
    ("snowball",    int(h * 0.80), int(h * 1.00), False)
]

for name, y1, y2, is_golden in rows:
    # Silahların bulunduğu orta şerit
    strip = img[y1:y2, int(w * 0.28):int(w * 0.72)]
    
    if is_golden:
        # Sari tabanca icin HSV esigi
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20, 100, 150]), np.array([35, 255, 255]))
    else:
        # Bembeyaz silah pikselleri
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 60:
            x, y, cw, ch = cv2.boundingRect(c)
            tight_crop = strip[y:y+ch, x:x+cw]
            tight_mask = mask[y:y+ch, x:x+cw]
            
            b, g, r = cv2.split(tight_crop)
            rgba = cv2.merge([b, g, r, tight_mask])
            
            out_path = os.path.join('templates', f"{name}.png")
            cv2.imwrite(out_path, rgba)
            print(f"Cikarildi: {out_path} ({cw}x{ch} px)")

print("Tum Valorant silahlari tamamlandi!")