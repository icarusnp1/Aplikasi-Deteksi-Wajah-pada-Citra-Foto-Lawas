import dlib
import cv2
import numpy as np

# Inisialisasi detector wajah
detector = dlib.get_frontal_face_detector()

# Baca gambar berwarna
image = cv2.imread("input/wajah4.jpg")
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan atau path salah")

# Deteksi wajah (skala 1)
faces = detector(image, 1)
if len(faces) == 0:
    raise ValueError("Wajah tidak terdeteksi dalam gambar")

# Ambil wajah pertama yang terdeteksi
face = faces[0]
x, y, w, h = face.left(), face.top(), face.width(), face.height()

# Crop wajah dan resize jadi 64x64 grayscale
out = image[y:y + h, x:x + w]
out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
out = cv2.resize(out, (64, 64))

print("Ukuran crop dan resize wajah:", out.shape)

# Ambil 1 cell 10x10 dari pojok kiri atas
cell = out[0:10, 0:10]
print("\nCell (10x10) dari pojok kiri atas:\n", cell)

# Kernel Sobel (versi klasik jadul)
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
Gy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]], dtype=np.float32)

H, W = cell.shape
gx = np.zeros((H - 2, W - 2), dtype=np.float32)
gy = np.zeros((H - 2, W - 2), dtype=np.float32)

# Konvolusi Sobel manual tanpa padding → output 8x8
for i in range(H - 2):
    for j in range(W - 2):
        region = cell[i:i+3, j:j+3]
        gx[i, j] = np.sum(region * Gx)
        gy[i, j] = np.sum(region * Gy)

print("\nHasil konvolusi gx:\n", gx)
print("\nHasil konvolusi gy:\n", gy)

# Hitung magnitude dan sudut gradien (angle 0-180)
magnitude = np.sqrt(gx**2 + gy**2)
angle = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

print("\nMagnitude gradien:\n", magnitude)
print("\nSudut gradien (derajat):\n", angle)

# Histogram soft binning dengan 9 bin
bins = 9
hist = np.zeros(bins, dtype=np.float32)
bin_width = 180 / bins

for i in range(magnitude.shape[0]):
    for j in range(magnitude.shape[1]):
        mag = magnitude[i, j]
        ang = angle[i, j]

        bin_pos = ang / bin_width
        left_bin = int(np.floor(bin_pos)) % bins
        right_bin = (left_bin + 1) % bins

        right_weight = bin_pos - left_bin
        left_weight = 1 - right_weight

        hist[left_bin] += mag * left_weight
        hist[right_bin] += mag * right_weight

print("\nHistogram orientasi (soft binning):\n", hist)
