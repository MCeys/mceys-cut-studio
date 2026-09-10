import os
import urllib.request
import cv2
import numpy as np

url = "https://preview.redd.it/tf41pllg81e91.png?width=1080&crop=smart&auto=webp&s=6f2441cfdf96b2be4ee99c855a805f15d2a71dc1"

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req)
    image_bytes = bytearray(resp.read())
    img = cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
except Exception:
    img = cv2.imread('killfeed.webp') or cv2.imread('killfeed.jpg') or cv2.imread('killfeed.png')

if img is None:
    print("Görsel yüklenemedi!")
    exit()

h, w, _ = img.shape

# En alttaki 3. şerit: Operator bölgesi (Yükseklik %70-%98, Genişlik %38-%56 arası)
op_crop = img[int(h*0.70):int(h*0.98), int(w*0.38):int(w*0.56)]

gray = cv2.cvtColor(op_crop, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    op_tight = op_crop[y:y+ch, x:x+cw]
    mask_tight = mask[y:y+ch, x:x+cw]
else:
    op_tight = op_crop
    mask_tight = mask

b, g, r = cv2.split(op_tight)
rgba = cv2.merge([b, g, r, mask_tight])

os.makedirs('templates', exist_ok=True)
output_path = os.path.join('templates', 'operator.png')
cv2.imwrite(output_path, rgba)
print(f"BAŞARILI! '{output_path}' oluşturuldu. Boyut: {rgba.shape[1]}x{rgba.shape[0]} px")