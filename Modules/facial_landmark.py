import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
image = cv2.imread("input/wajah4.jpg")
faces = detector(image, 1)

for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    out = image[y:y + h, x:x + w]

out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
out = cv2.resize(out, (64, 64))  # Resize ke 10x10 piksel

# H, W = out.shape[:2]  # Ambil ukuran dari wajah yang terdeteksi
# for h in range(H):
#     for w in range(W):
#         print(f"{h} {w} = {out[h, w]}")

H, W = out.shape[:2]  # Ambil ukuran dari wajah yang terdeteksi

dummy = np.zeros((H - (H - 10), W - (W - 10)), dtype=np.uint8)
for h in range(H - (H - 10)):
    for w in range(W  - (W - 10)):
        print(f"{h} {w} = {out[h, w]}")
        dummy[h, w] = out[h, w]

# out = cv2.resize(out, (1000, 1000))  # Resize ke 10x10 piksel

# H, W = out.shape[:2]  # Ambil ukuran dari wajah yang terdeteksi
# for h in range(H):
#     for w in range(W):
#         print(f"{h} {w} = {out[h, w]}")
        
# dummy = cv2.resize(dummy, (300, 600))  # Resize ke 10x10 piksel
cv2.imshow("Wajah2", dummy)
# cv2.imshow("Hasil", image)
# cv2.imwrite("output/wajah4.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()