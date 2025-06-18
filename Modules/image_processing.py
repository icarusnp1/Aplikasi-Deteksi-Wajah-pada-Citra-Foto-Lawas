import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

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

    plt.show()

def detect_faces(image, predictor_path='Modules/shape_predictor_68_face_landmarks.dat', max_faces=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Pastikan gambar dalam format RGB uint8
    if len(image.shape) == 2:
        input_for_dlib = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        draw_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            input_for_dlib = image
            draw_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Image harus bertipe uint8")
    else:
        raise ValueError("Format gambar tidak valid untuk Dlib")

    faces = detector(input_for_dlib, 1)
    if max_faces is not None:
        faces = faces[:max_faces]

    for face in faces:
        landmarks = predictor(input_for_dlib, face)

        # Landmark penting
        p48 = landmarks.part(48)
        p54 = landmarks.part(54)
        p51 = landmarks.part(51)
        p57 = landmarks.part(57)

        # Mulut (48-67), mata kiri (36-41), mata kanan (42-47)
        mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

        # Eye Aspect Ratio (EAR)
        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Titik y penting
        left_corner_y = p48.y
        right_corner_y = p54.y
        top_lip_y = p51.y
        bottom_lip_y = p57.y

        # Threshold
        EAR_SLEEP = 0.20
        SMILE_CORNER_DIFF = 10
        MOUTH_OPEN_THRESHOLD = 8

        left_corner_diff = bottom_lip_y - left_corner_y
        right_corner_diff = bottom_lip_y - right_corner_y
        mouth_distance = abs(bottom_lip_y - top_lip_y)

        # Deteksi senyum
        is_smile = (
            left_corner_diff >= SMILE_CORNER_DIFF and
            right_corner_diff >= SMILE_CORNER_DIFF and
            top_lip_y < bottom_lip_y
        )

        # Deteksi mulut terbuka
        mouth_open = mouth_distance > MOUTH_OPEN_THRESHOLD

        # Ekspresi akhir
        if avg_ear < EAR_SLEEP:
            expression = "Ngantuk"
        elif is_smile:
            expression = "Senyum"
        elif not mouth_open:
            expression = "Netral"
        else:
            expression = "Netral"

        # === Cetak ke TERMINAL ===
        print(f"\nTitik 48 (sudut kiri)        : (x={p48.x}, y={p48.y})")
        print(f"Titik 54 (sudut kanan)       : (x={p54.x}, y={p54.y})")
        print(f"Titik 51 (bibir atas tengah) : (x={p51.x}, y={p51.y})")
        print(f"Titik 57 (bibir bawah tengah): (x={p57.x}, y={p57.y})")

        print("\nPerhitungan:")
        print(f"Jarak bibir: |{p51.y} - {p57.y}| = {mouth_distance} piksel")

        print(f"\nSudut kiri & kanan mulut: y={p48.y} & y={p54.y}")
        if p48.y < p57.y and p54.y < p57.y:
            print(f"➤ Keduanya lebih tinggi dari bibir bawah (y={p57.y}) ✅")
        else:
            print(f"➤ Salah satu atau kedua sudut tidak lebih tinggi dari bibir bawah ❌")

        print("\n➡️ Karena:")
        print(f"Jarak mulut > {MOUTH_OPEN_THRESHOLD} piksel {'✅' if mouth_open else '❌'}")
        if p48.y < p57.y and p54.y < p57.y:
            print("Sudut bibir lebih tinggi dari bawah ✅")
        else:
            print("Sudut bibir lebih tinggi dari bawah ❌")

        if mouth_open and is_smile:
            print("\n➡️ Maka kondisi mulut: tersenyum dan terbuka 😄")
        elif mouth_open and not is_smile:
            print("\n➡️ Maka kondisi mulut: terbuka tanpa senyum 😐")
        elif not mouth_open and is_smile:
            print("\n➡️ Maka kondisi mulut: tersenyum tapi tidak terbuka 😊")
        else:
            print("\n➡️ Maka kondisi mulut: netral 😐")

        # === Visualisasi di Gambar ===
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(draw_img, expression, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i in range(68):
            px = landmarks.part(i).x
            py = landmarks.part(i).y
            cv2.circle(draw_img, (px, py), 2, (255, 0, 0), -1)
            cv2.putText(draw_img, str(i), (px + 2, py - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)

    # --- Info Ringkas di Gambar ---
    total_faces = len(faces)
    cv2.putText(draw_img, f"Total Wajah: {total_faces}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if total_faces > 0:
        info_y = 55
        spacing = 23
        info_lines = [
            f"Left EAR      : {left_ear:.2f}",
            f"Right EAR     : {right_ear:.2f}",
            f"Avg EAR       : {avg_ear:.2f}",
            f"Titik 51 (atas bibir)  : y = {top_lip_y}",
            f"Titik 57 (bawah bibir) : y = {bottom_lip_y}",
            f"Jarak bibir    : {mouth_distance:.2f} px {'> threshold' if mouth_open else '<= threshold'}",
            f"Titik 48 y={left_corner_y} dan 54 y={right_corner_y}",
            f"Selisih kiri: {left_corner_diff}, kanan: {right_corner_diff}",
            f"→ Smile: {'Ya' if is_smile else 'Tidak'}",
            f"Mulut Terbuka: {'Ya' if mouth_open else 'Tidak'}",
            f"Ekspresi: {expression}",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(draw_img, line, (10, info_y + spacing * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Konversi BGR ke RGB jika pakai matplotlib
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    return draw_img

def restore_face_color(image, predictor_path='Modules/shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    # Deteksi wajah dengan dlib pada gambar RGB
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)

    if len(faces) == 0:
        print("Tidak ada wajah terdeteksi.")
        return image

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y = max(x, 0), max(y, 0)
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