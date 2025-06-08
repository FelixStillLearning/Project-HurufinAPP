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
        """Terapkan operasi morphological opening (erosi diikuti dilasi) secara manual."""
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        # Operasi opening adalah erosi diikuti dilasi
        self.morphological_erosion(kernel_size, iterations=1)
        self.morphological_dilation(kernel_size, iterations=1)
        
        return self.processed_image
    
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
        """Konversi gambar ke biner berdasarkan threshold tertentu."""
        h, w = img.shape[:2]
        binary_img = np.zeros((h, w), np.uint8)

        for i in range(h):
            for j in range(w):
                binary_img[i, j] = 255 if img[i, j] > threshold else 0

        return binary_img
    
    def gaussian_blur(self, kernel_size=3, sigma=1.0):
        """Terapkan filter Gaussian blur secara manual.
        
        Args:
            kernel_size (int): Ukuran kernel (akan dibuat kernel_size x kernel_size).
            sigma (float): Standar deviasi distribusi Gaussian.
        
        Returns:
            numpy.ndarray: Gambar hasil filter Gaussian blur.
        """
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image
        
        # Pastikan gambar sudah grayscale
        if len(img.shape) == 3:
            img = self.grayscale()
        
        # Buat kernel Gaussian
        k = kernel_size // 2
        x, y = np.mgrid[-k:k+1, -k:k+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalisasi kernel
        
        h, w = img.shape
        padding = kernel_size // 2
        
        # Buat gambar dengan padding
        padded_img = np.zeros((h + 2 * padding, w + 2 * padding), np.uint8)
        padded_img[padding:padding + h, padding:padding + w] = img
        
        # Lakukan konvolusi
        output = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(
                    padded_img[i:i + kernel_size, j:j + kernel_size] * kernel
                )
        
        self.processed_image = output.astype(np.uint8)
        return self.processed_image
    
    def morphological_erosion(self, kernel_size=(3, 3), iterations=1):
        """Terapkan operasi morphological erosion secara manual.
        
        Args:
            kernel_size (tuple): Ukuran kernel (height, width).
            iterations (int): Jumlah iterasi operasi erosi.
            
        Returns:
            numpy.ndarray: Gambar hasil erosi.
        """
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image.copy()
        
        # Pastikan gambar sudah grayscale atau biner
        if len(img.shape) == 3:
            img = self.grayscale()
        
        # Gunakan cv2 untuk membuat structuring element
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        # Ukuran kernel dan dimensi gambar
        h, w = img.shape
        kh, kw = strel.shape
        
        # Padding untuk kernel
        pad_h = kh // 2
        pad_w = kw // 2
        
        # Lakukan erosi untuk sejumlah iterasi
        result = img.copy()
        
        for _ in range(iterations):
            temp = np.zeros_like(result)
            
            # Padding gambar
            padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
            
            for i in range(h):
                for j in range(w):
                    # Ambil region berdasarkan ukuran kernel
                    region = padded[i:i + kh, j:j + kw]
                    
                    # Erosi: Pixel bernilai 1 (255) hanya jika semua pixel yang tumpang tindih dengan kernel bernilai 1
                    if np.all(np.logical_or(region == 255, strel == 0)):
                        temp[i, j] = 255
                    else:
                        temp[i, j] = 0
            
            result = temp
        
        self.processed_image = result
        return self.processed_image
    
    def morphological_dilation(self, kernel_size=(3, 3), iterations=1):
        """Terapkan operasi morphological dilation secara manual.
        
        Args:
            kernel_size (tuple): Ukuran kernel (height, width).
            iterations (int): Jumlah iterasi operasi dilasi.
            
        Returns:
            numpy.ndarray: Gambar hasil dilasi.
        """
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image.copy()
        
        # Pastikan gambar sudah grayscale atau biner
        if len(img.shape) == 3:
            img = self.grayscale()
        
        # Gunakan cv2 untuk membuat structuring element
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        # Ukuran kernel dan dimensi gambar
        h, w = img.shape
        kh, kw = strel.shape
        
        # Padding untuk kernel
        pad_h = kh // 2
        pad_w = kw // 2
        
        # Lakukan dilasi untuk sejumlah iterasi
        result = img.copy()
        
        for _ in range(iterations):
            temp = np.zeros_like(result)
            
            # Padding gambar dengan nilai 0 (latar belakang)
            padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
            
            for i in range(h):
                for j in range(w):
                    # Ambil region berdasarkan ukuran kernel
                    region = padded[i:i + kh, j:j + kw]
                    
                    # Dilasi: Pixel bernilai 1 (255) jika ada pixel yang tumpang tindih dengan kernel bernilai 1
                    if np.any(np.logical_and(region == 255, strel == 1)):
                        temp[i, j] = 255
                    else:
                        temp[i, j] = 0
            
            result = temp
        
        self.processed_image = result
        return self.processed_image
    
    def invert_binary(self):
        """Balikkan piksel gambar biner secara manual (0 menjadi 255, 255 menjadi 0).
        
        Returns:
            numpy.ndarray: Gambar biner yang dibalik.
        """
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image.copy()
        
        # Pastikan gambar sudah grayscale
        if len(img.shape) == 3:
            img = self.grayscale()
        
        h, w = img.shape
        inverted = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                inverted[i, j] = 255 - img[i, j]
        
        self.processed_image = inverted
        return self.processed_image
    
    def morphological_closing(self, kernel_size=(3, 3)):
        """Terapkan operasi morphological closing secara manual (dilasi diikuti erosi).
        
        Args:
            kernel_size (tuple): Ukuran kernel (height, width).
            
        Returns:
            numpy.ndarray: Gambar hasil closing.
        """
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        # Operasi closing adalah dilasi diikuti erosi
        self.morphological_dilation(kernel_size, iterations=1)
        self.morphological_erosion(kernel_size, iterations=1)
        
        return self.processed_image
    
    def adaptive_threshold(self, block_size=11, c=2):
        """Terapkan adaptive thresholding secara manual.
        
        Args:
            block_size (int): Ukuran blok untuk menghitung nilai threshold lokal (harus ganjil).
            c (int): Konstanta yang dikurangkan dari rata-rata atau weighted mean.
            
        Returns:
            numpy.ndarray: Gambar hasil adaptive thresholding.
        """
        if self.processed_image is None:
            raise ValueError("Belum ada gambar yang dimuat")
        
        img = self.processed_image.copy()
        
        # Pastikan gambar sudah grayscale
        if len(img.shape) == 3:
            img = self.grayscale()
        
        if block_size % 2 == 0:
            block_size += 1  # Pastikan ukuran blok ganjil
        
        h, w = img.shape
        output = np.zeros_like(img)
        
        # Padding
        pad_size = block_size // 2
        padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), 
                           mode='constant', constant_values=0)
        
        # Lakukan adaptive thresholding
        for i in range(h):
            for j in range(w):
                # Hitung nilai threshold lokal (rata-rata dalam blok)
                block = padded_img[i:i + block_size, j:j + block_size]
                threshold = np.mean(block) - c
                
                # Terapkan threshold
                if img[i, j] > threshold:
                    output[i, j] = 0  # Inverse untuk mendapatkan THRESH_BINARY_INV
                else:
                    output[i, j] = 255
        
        self.processed_image = output
        return self.processed_image