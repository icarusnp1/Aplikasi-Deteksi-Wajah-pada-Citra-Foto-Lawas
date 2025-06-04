import cv2
import numpy as np

# Baca dan ubah ke grayscale
image = cv2.imread('input/wajah4.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ambil ukuran
H, W = image_gray.shape
out = np.zeros((10, 10), dtype=np.uint8)  # hanya 10x10 piksel pertama

# Ambil dan cetak nilai grayscale
for i in range(10):
    for j in range(10):
        out[i, j] = image_gray[i, j]
        print(f"{i} {j} = {out[i,j]}")  # Nilai pixel grayscale

# Tampilkan gambar hasil
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
