import dlib
import cv2
import numpy as np

def sobel_gradients(image):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    gx = cv2.filter2D(image, -1, Gx)
    gy = cv2.filter2D(image, -1, Gy)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180) % 180
    return magnitude, angle

def cell_histogram(mag, ang, bins=9):
    hist = np.zeros(bins, dtype=np.float32)
    bin_width = 180 / bins
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            a = ang[i, j]
            m = mag[i, j]
            bin_pos = a / bin_width
            l_bin = int(np.floor(bin_pos)) % bins
            r_bin = (l_bin + 1) % bins
            r_w = bin_pos - l_bin
            l_w = 1 - r_w
            hist[l_bin] += m * l_w
            hist[r_bin] += m * r_w
    return hist

def normalize_block(block_hist):
    norm = np.sqrt(np.sum(block_hist**2) + 1e-5)
    return block_hist / norm

def extract_blocks(image, cell_size=10):
    blocks = []
    for by in range(0, 64 - 20 + 1, 20):
        for bx in range(0, 64 - 20 + 1, 20):
            block = []
            for dy in [0, 10]:
                for dx in [0, 10]:
                    y0, y1 = by + dy, by + dy + cell_size
                    x0, x1 = bx + dx, bx + dx + cell_size
                    if y1 <= image.shape[0] and x1 <= image.shape[1]:
                        block.append(image[y0:y1, x0:x1])
            if len(block) == 4:
                blocks.append(block)
    return blocks

# Load image dan deteksi wajah
detector = dlib.get_frontal_face_detector()
image = cv2.imread("input/wajah4.jpg")
faces = detector(image, 1)

if not faces:
    print("Wajah tidak ditemukan.")
    exit()

x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
face_img = image[y:y+h, x:x+w]
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (64, 64))

# Ekstraksi blok
blocks = extract_blocks(gray)
if not blocks:
    print("Tidak ada blok valid.")
    exit()

block0 = blocks[0]
block0_hist = []

print("📦 Detail Blok #0:")

for i, cell in enumerate(block0):
    mag, ang = sobel_gradients(cell)
    hist = cell_histogram(mag, ang)
    block0_hist.extend(hist)

    print(f"\n🔹 Cell #{i}")
    print("Magnitude:\n", np.round(mag, 2))
    print("Angle:\n", np.round(ang, 1))
    print("Histogram:", np.round(hist, 2))

# Normalisasi blok 0
block0_hist = np.array(block0_hist, dtype=np.float32)
norm_block0 = normalize_block(block0_hist)

print("\n✅ Histogram Blok #0 Setelah Normalisasi:")
print(np.round(norm_block0, 4))
print("Panjang descriptor:", len(norm_block0))
