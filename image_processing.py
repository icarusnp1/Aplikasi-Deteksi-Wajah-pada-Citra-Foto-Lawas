import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def enhance_image(image_path, gVal, clVal):
    # Baca gambar
    img = cv2.imread(image_path)

    # Mengubah ke format LAB untuk meningkatkan kontras
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Pisahkan channel L, A, dan B
    l, a, b = cv2.split(lab)

    # Gunakan CLAHE untuk meningkatkan kontras di channel L
    clahe = cv2.createCLAHE(clipLimit=gVal, tileGridSize=(clVal, clVal))
    l = clahe.apply(l)

    # Gabungkan kembali channel
    lab = cv2.merge((l, a, b))

    # Konversi kembali ke BGR
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Tampilkan histogram
    plot_rgb_histogram(img, enhanced_img, "Enhancement")

    return enhanced_img


def denoise_image(img, hs, hcs, sws, tws):
    original_img = img.copy()

    # Lakukan denoising
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, hs, hcs, sws, tws)

    # Tampilkan histogram
    plot_rgb_histogram(original_img, denoised_img, "Denoising")

    return denoised_img


def sharpen_image(image, k=1):
    # Kernel sharpening dasar
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + k, -1],
                       [0, -1, 0]])
    
    # Konvolusi
    sharpened = cv2.filter2D(image, -1, kernel)

    # Pastikan hasil tetap dalam range 0-255
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def restore_color(img, sScale):
    # Saturation scale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sScale, 0, 255)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    restored_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Tampilkan histogram
    plot_rgb_histogram(img, restored_img, "Color Restoration")

    return restored_img


def Grayscale(img):
    # Mengonversi gambar ke grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tampilkan histogram untuk gambar grayscale
    plot_gray_histogram(img, gray_image)

    return gray_image


def Biner(img):
    # Mengonversi gambar ke grayscale terlebih dahulu
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buat gambar biner berdasarkan threshold
    _, biner = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

    # Tampilkan histogram untuk gambar biner
    plot_gray_histogram(img, biner)

    return biner


def plot_rgb_histogram(original_img, processed_img, title):
    # Hitung total nilai RGB untuk gambar asli
    original_red = np.sum(original_img[:, :, 0])
    original_green = np.sum(original_img[:, :, 1])
    original_blue = np.sum(original_img[:, :, 2])

    # Hitung total nilai RGB untuk gambar yang diproses
    processed_red = np.sum(processed_img[:, :, 0])
    processed_green = np.sum(processed_img[:, :, 1])
    processed_blue = np.sum(processed_img[:, :, 2])

    # Buat histogram
    labels = ['Red', 'Green', 'Blue']
    original_values = [original_red, original_green, original_blue]
    processed_values = [processed_red, processed_green, processed_blue]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # lebar bar

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, original_values, width, label='Gambar Asli')
    bars2 = ax.bar(x + width / 2, processed_values, width, label='Gambar Diproses')

    # Tambahkan beberapa teks untuk label, title dan custom x-axis tick labels, etc.
    ax.set_ylabel('Total Nilai RGB')
    ax.set_title(f'Total Nilai RGB dari Gambar Asli dan {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Tampilkan histogram
    plt.show()


def plot_gray_histogram(original_img, processed_img):
    # Hitung total nilai untuk gambar asli
    original_gray = np.sum(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY))

    # Hitung total nilai untuk gambar biner
    processed_gray = np.sum(processed_img)

    # Buat histogram
    labels = ['Grayscale', 'Biner']
    values = [original_gray, processed_gray]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # lebar bar

    fig, ax = plt.subplots()
    bars = ax.bar(x, values, width, label='Total Nilai')

    # Tambahkan beberapa teks untuk label, title dan custom x-axis tick labels, etc.
    ax.set_ylabel('Total Nilai Grayscale')
    ax.set_title('Total Nilai Grayscale dan Biner')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Tampilkan histogram
    plt.show()

def detect_faces(im, PREDICTOR_PATH):
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    detector = dlib.get_frontal_face_detector()

    class TooManyFaces(Exception):
        pass

    class NoFaces(Exception):
        pass

    def get_landmarks_and_rect(im):
        rects = detector(im, 1)
        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]), rects[0]

    def annotate_landmarks_and_box(im, landmarks, rect):
        im = im.copy()
        # Gambar landmark
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        # Gambar kotak di sekitar wajah
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return im

    try:
        landmarks, rect = get_landmarks_and_rect(im)
        image_with_landmarks = annotate_landmarks_and_box(im, landmarks, rect)
        return image_with_landmarks
    except TooManyFaces:
        print("Terdeteksi lebih dari satu wajah.")
        return im
    except NoFaces:
        print("Tidak ada wajah terdeteksi.")
        return im

def restore_face_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("Tidak ada wajah terdeteksi.")
        return image

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]

        # Convert ke LAB untuk restorasi warna
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        restored_face = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Tempel hasil kembali ke gambar asli
        image[y:y+h, x:x+w] = restored_face

    return image

