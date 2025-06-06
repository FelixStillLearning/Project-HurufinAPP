# Modul preprocessing gambar untuk aplikasi Hurufin OCR
"""
Modul preprocessing gambar untuk aplikasi Hurufin OCR.
Berisi semua operasi pemrosesan gambar seperti konversi grayscale,
pengurangan noise, contrast stretching, thresholding, dan operasi morfologi.
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Kelas untuk menangani operasi preprocessing gambar untuk OCR.
    Semua proses ada di sini :)
    """
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
    def load_image(self, image_path):
        """Muat gambar dari file path."""
        self.original_image = cv2.imread(image_path)
        self.processed_image = self.original_image.copy()
        return self.original_image is not None
    
    def set_image(self, image):
        """Set gambar langsung dari numpy array."""
        self.processed_image = image.copy()
    
    def grayscale(self):
        """Konversi gambar ke grayscale dengan implementasi manual."""
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image
        h, w = img.shape[:2]
        gray = np.zeros((h, w), np.uint8)

        for i in range(h):
            for j in range(w):
                # Rumus grayscale
                gray[i, j] = np.clip(0.299 * img[i, j, 0] + 
                                   0.587 * img[i, j, 1] + 
                                   0.114 * img[i, j, 2], 0, 255)
        
        self.processed_image = gray
        return gray

    def median_blur(self, kernel_size=3):
        """Terapkan filter median blur untuk mengurangi noise."""
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image

        # Pastikan gambar sudah grayscale
        if len(img.shape) == 3:
            img = self.grayscale()

        h, w = img.shape[:2]
        pad_size = kernel_size // 2

        # Buat gambar dengan padding
        padded_img = np.zeros((h + 2 * pad_size, w + 2 * pad_size), np.uint8)
        padded_img[pad_size:pad_size + h, pad_size:pad_size + w] = img

        filtered_img = np.zeros_like(img)
        for i in range(pad_size, h + pad_size):
            for j in range(pad_size, w + pad_size):
                region = padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                filtered_img[i - pad_size, j - pad_size] = np.median(region)

        self.processed_image = filtered_img
        return filtered_img
    
    def contrast_stretching(self):
        """Terapkan contrast stretching untuk meningkatkan kontras gambar."""
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        img = self.processed_image
        h, w = img.shape[:2]
        minV = np.min(img)
        maxV = np.max(img)

        stretched_img = img.copy()
        for i in range(h):
            for j in range(w):
                a = img[i, j]
                b = float(a - minV) / (maxV - minV) * 255
                stretched_img[i, j] = b

        self.processed_image = stretched_img
        return stretched_img
    
    def otsu_threshold(self):
        """Terapkan metode thresholding Otsu."""
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image

        if len(img.shape) == 3:
            img = self.grayscale()

        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        variances = np.zeros(256)

        for t in range(256):
            w1 = np.sum(hist[:t+1])
            w2 = np.sum(hist[t+1:])            
            if w1 == 0 or w2 == 0:
                continue
            m1 = np.sum(np.arange(t + 1) * hist[:t + 1]) / w1
            m2 = np.sum(np.arange(t + 1, 256) * hist[t + 1:]) / w2
            variances[t] = float(w1) * float(w2) * ((m1 - m2) ** 2)
        
        threshold = np.argmax(variances)
        binary_img = self.binary(img, threshold)

        self.processed_image = binary_img
        return binary_img

    def morphological_opening(self, kernel_size=(3, 3)):
        """Terapkan operasi morphological opening."""
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image

        if len(img.shape) == 3:
            img = self.grayscale()

        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, strel)

        self.processed_image = opened_img
        return opened_img
    
    def preprocess_pipeline(self):
        """
        Pipeline preprocessing lengkap:
        1. Konversi grayscale
        2. Median blur
        3. Contrast stretching
        4. Otsu thresholding
        5. Morphological opening
        """
        try:
            # Langkah 1: Grayscale
            self.grayscale()
            
            # Langkah 2: Median Blur
            self.median_blur(kernel_size=3)
            
            # Langkah 3: Contrast Stretching
            self.contrast_stretching()
            
            return self.processed_image
            
        except Exception as e:
            print(f"Error pada pipeline preprocessing: {e}")
            return None
    
    def get_processed_image(self):
        """Ambil gambar hasil proses saat ini."""
        return self.processed_image
    
    def get_original_image(self):
        """Ambil gambar asli."""
        return self.original_image    
    
    def binary(self, img, threshold=127):

        h, w = img.shape[:2]
        binary_img = np.zeros((h, w), np.uint8)

        for i in range(h):
            for j in range(w):
                binary_img[i, j] = 255 if img[i, j] > threshold else 0

        return binary_img