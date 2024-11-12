import sys
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5 import QtCore
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class ChessPieceDetector:
    def __init__(self, dataset_path, target_size=(128, 128)):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
        self.svm_model = SVC(kernel='linear', probability=True)
        self.label_map = {'pawn_resized': 'pawn', 'Rook-resize': 'rook', 'knight-resize': 'knight',
                          'bishop_resized': 'bishop', 'Queen-Resized': 'queen', 'King_resize': 'king'}
        self.train()

    def load_and_preprocess_images(self):
        images, labels = [], []
        for label_name in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label_name)
            if not os.path.isdir(label_path):
                continue
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Error: Gambar tidak ditemukan atau tidak bisa dibuka - {image_path}")
                    continue
                image = cv2.resize(image, self.target_size)

                images.append(image)
                labels.append(self.label_map.get(label_name, label_name))

                images.append(cv2.flip(image, 1))
                labels.append(self.label_map.get(label_name, label_name))
        return np.array(images), np.array(labels)

    def extract_hog_features(self, images):
        hog_features = [hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                        for image in images]
        return np.array(hog_features)

    def train(self):
        images, labels = self.load_and_preprocess_images()
        hog_features = self.extract_hog_features(images)
        labels_encoded = self.label_encoder.fit_transform(labels)

        x_train, x_test, y_train, y_test = train_test_split(hog_features, labels_encoded, test_size=0.2,
                                                            random_state=42)

        self.svm_model.fit(x_train, y_train)

        y_train_pred = self.svm_model.predict(x_train)
        print(f'Training Accuracy: {accuracy_score(y_train, y_train_pred)}')

        y_pred = self.svm_model.predict(x_test)

        print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}')


    def detect_from_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # membaca path gambar dengan format berwarna
        if image is None:
            print("Error: Gambar tidak ditemukan atau tidak bisa dibuka.") # penanganan jika gambar tidak ditemukan
            return None, None

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # proses merubah gambar menjadi skala keabuan
        image_resized = cv2.resize(image_gray, self.target_size) # merubah ukuran hasil gambar grayscale

        # Menampilkan hasil gambar yang telah melalui proses grayscale dan resize
        self.display_grayscale(image_gray)
        self.display_resized(image_resized)

        # Ekstraksi fitur HOG
        hog_features, hog_image = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

        # Menggunakan fitur HOG untuk prediksi
        features = hog_features.reshape(1, -1)
        prediction = self.svm_model.predict(features)
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]

        # Visualisasi fitur HOG
        self.display_hog(hog_image)

        # Determine font scale and thickness based on image dimensions
        font_scale = min(image.shape[1] / 600, 1)
        thickness = max(int(image.shape[1] / 300), 1)

        cv2.putText(image, f'Detected: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                    thickness, cv2.LINE_AA)
        return predicted_label, image

    def display_grayscale(self, image):
        # Menampilkan gambar grayscale menggunakan Matplotlib
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        plt.show()

    def display_resized (self, image):
        # Menampilkan visualisasi resized Matplotlib
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title('Resized Image')
        plt.axis('off')
        plt.show()

    def display_hog (self, image):
        # Menampilkan visualisasi fitur HOG menggunakan Matplotlib
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title('HOG Feature Visualization')
        plt.axis('off')
        plt.show()

    def display_random_hog(self):
        fig, axes = plt.subplots(nrows=len(self.label_map), ncols=2, figsize=(10, 10))
        fig.tight_layout()

        for i, (label_name, label) in enumerate(self.label_map.items()):
            label_path = os.path.join(self.dataset_path, label_name)
            if not os.path.isdir(label_path):
                continue
            images = [img for img in os.listdir(label_path) if img.endswith(".jpg")]
            img_name = np.random.choice(images)
            image_path = os.path.join(label_path, img_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_resized = cv2.resize(image, self.target_size)
            hog_features, hog_image = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

            print(f"HOG features for {label}:", hog_features)

            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title(f"Original - {label}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(hog_image, cmap='gray')
            axes[i, 1].set_title(f"HOG - {label}")
            axes[i, 1].axis('off')



        plt.show()

class ShowImage(QDialog):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('gui.ui', self)

        self.detector = ChessPieceDetector(
            r'C:\SEMESTER 4\PENGOLAHAN CITRA DIGITAL\PRAKTIKUM\E6_Aplikasi\Chess_Piece')

        self.pushButton = self.findChild(QPushButton, 'pushButton')
        self.pushButton_2 = self.findChild(QPushButton, 'pushButton_2')
        self.pushButton_4 = self.findChild(QPushButton, 'pushButton_4')
        self.imgLabel = self.findChild(QLabel, 'imgLabel')
        self.imgLabel_2 = self.findChild(QLabel, 'imgLabel_2')

        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.process_image)
        self.pushButton_4.clicked.connect(self.detector.display_random_hog)

        self.filePath = None

    def load_image(self):
        self.filePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if self.filePath:
            self.display_image(self.filePath, 1)

    def process_image(self):
        if self.filePath:
            detected_label, processed_image = self.detector.detect_from_image(self.filePath)
            if detected_label:
                self.display_image(processed_image, 2)

    def display_image(self, image_or_path, window):
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qImg)
        if window == 1:
            self.imgLabel.setPixmap(
                pixmap.scaled(self.imgLabel.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.imgLabel.setScaledContents(False)
        elif window == 2:
            self.imgLabel_2.setPixmap(
                pixmap.scaled(self.imgLabel_2.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.imgLabel_2.setAlignment(QtCore.Qt.AlignCenter)
            self.imgLabel_2.setScaledContents(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShowImage()
    window.show()
    sys.exit(app.exec_())
