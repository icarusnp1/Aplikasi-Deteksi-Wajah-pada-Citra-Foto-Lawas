from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import sys
import cv2
from Modules.image_processing import enhance_image, denoise_image, sharpen_image, restore_color, Biner, Grayscale
import numpy as np
# fungsi face detection
from Modules.image_processing import detect_faces
from Modules.image_processing import restore_face_color

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("Modules\GUI.ui", self)
        
        self.image_to_save = None

        self.image_label1 = QtWidgets.QLabel(self.scrollArea1)
        self.scrollArea1.setWidget(self.image_label1)
        self.scrollArea1.setWidgetResizable(True)

        self.image_label2 = QtWidgets.QLabel(self.scrollArea2)
        self.scrollArea2.setWidget(self.image_label2)
        self.scrollArea2.setWidgetResizable(True)

        self.loadBut.clicked.connect(self.loadImage)
        self.denoiseBut.clicked.connect(self.apply_denoise)
        self.sharpBut.clicked.connect(self.apply_sharpen)
        self.resBut.clicked.connect(self.apply_restore)
        self.fixBut.clicked.connect(self.apply_fiximage)
        self.saveBut.clicked.connect(self.saveImage)

        self.comboBox.currentTextChanged.connect(self.comboBox_)
        self.detectBut.clicked.connect(self.apply_face_detection)
        self.restoreFaceColorBut.clicked.connect(self.apply_face_color_restore)


    def comboBox_(self, text):
        if text == "Biner":
            self.apply_biner()
        elif text == "Grayscale":
            self.apply_grayscale()

    def loadImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.imagePath = fname
            self.original_image = cv2.imread(fname)

            pixmap = QPixmap(fname)
            self.image_label1.setPixmap(pixmap)
            self.image_label1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.image_label1.resize(pixmap.size())

    def saveImage(self):
        if self.image_to_save is not None:
            # Buka dialog untuk memilih lokasi penyimpanan
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                       "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
                                                       options=options)
            if file_name:
                # Simpan gambar
                img_to_save = cv2.cvtColor(self.image_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_name, img_to_save)
                print(f"Gambar disimpan di: {file_name}")
        else:
            print("Tidak ada gambar untuk disimpan.")

    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qImg)

        self.image_label2.setPixmap(pixmap)
        self.image_label2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.image_label2.resize(pixmap.size())
        # Reset image_to_save saat memuat gambar baru
        self.image_to_save = img

    def apply_fiximage(self):
        # grid
        # clip
        gVal = self.gValue.value()
        clVal = self.clValue.value()
        if self.imagePath:
            result = enhance_image(self.imagePath, gVal, clVal)
            self.displayImage(result)

    def apply_denoise(self):
        # image sharp
        # h sharp
        # hcolor sharp
        # template window sharp
        # search window sharp
        hs = self.hSharp.value()
        hcs = self.hSharp.value()
        tws = self.hSharp.value()
        sws = self.hSharp.value()
        if self.imagePath:
            result = denoise_image(cv2.imread(self.imagePath), hs, hcs, tws, sws)
            self.displayImage(result)

    def apply_sharpen(self):
        # Kernel value
        kValue = self.kValue.value()
        if self.imagePath:
            result = sharpen_image(cv2.imread(self.imagePath), kValue)
            self.displayImage(result)

    def apply_restore(self, sScale):
        # saturation scale
        if self.imagePath:
            sScale = self.sValue.value()
            result = restore_color(cv2.imread(self.imagePath), sScale)
            self.displayImage(result)

    def apply_biner(self):
        if self.imagePath:
            result = Biner(cv2.imread(self.imagePath))
            self.displayImage(result)

    def apply_grayscale(self):
        if self.imagePath:
            result = Grayscale(cv2.imread(self.imagePath))
            self.displayImage(result)

    def apply_face_detection(self):
        if self.imagePath:
            img = cv2.imread(self.imagePath)
            result = detect_faces(img)
            self.displayImage(result)
        else:
            print("Gambar belum dimuat.")

    def apply_face_color_restore(self):
        if self.imagePath:
            img = cv2.imread(self.imagePath)
            result = restore_face_color(img)
            self.displayImage(result)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())