"""
Kelas jendela utama untuk aplikasi Hurufin OCR.
Menangani antarmuka GUI dan mengintegrasikan fungsionalitas pemrosesan citra dan OCR.
"""

import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

# Tambahkan direktori induk ke dalam path untuk mengimpor modul kami
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_processing.preprocessing import ImagePreprocessor
from image_processing.ocr import OCRProcessor


class HurufinMainWindow(QMainWindow):
    """
    Kelas jendela utama untuk aplikasi Hurufin OCR.
    """
    
    def __init__(self):
        super(HurufinMainWindow, self).__init__()
        
        # Memuat file UI
        ui_path = os.path.join(os.path.dirname(__file__), "GUI.ui")
        loadUi(ui_path, self)
        
        # Inisialisasi pemroses
        self.preprocessor = ImagePreprocessor()
        self.ocr_processor = OCRProcessor()
        
        # Penyimpanan citra saat ini
        self.current_image = None
        
        # Penyimpanan untuk ML processing
        self.ml_preprocessed_image = None
        self.character_bboxes = []
        self.ml_model = None
        self.ml_detections = []
        
        # Hubungkan tombol ke metode
        self.setup_ml_connections()
    

    def setup_ml_connections(self):
        """Setup koneksi untuk ML processing buttons."""
        #Implementasi ini akan ditambahkan setelah buttons ML dibuat di GUI
        self.btn_input.clicked.connect(self.load_image)
        self.btn_step1.clicked.connect(self.apply_ml_preprocessing)
        self.btn_step2.clicked.connect(self.perform_character_segmentation) 
        self.btn_step3.clicked.connect(self.load_and_prepare_ml_model)
        self.btn_recognition.clicked.connect(self.perform_ml_character_recognition)
        self.btn_reset.clicked.connect(self.reset_ml_processing)
    
    def load_image(self):
        """Muat file gambar menggunakan file dialog."""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, 
            "Pilih File Gambar", 
            ".", 
            "File Gambar (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if image_path:
            success = self.preprocessor.load_image(image_path)
            if success:
                self.current_image = self.preprocessor.get_original_image()
                self.display_image(self.current_image, self.display_citra)
                print(f"Gambar berhasil dimuat: {os.path.basename(image_path)}")
            else:
                QMessageBox.warning(self, "Kesalahan", "Gagal memuat gambar!")
    
    def apply_preprocessing(self):
        """Terapkan pipeline pra-pemrosesan (grayscale, blur, contrast stretching)."""
        if self.current_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan muat gambar terlebih dahulu!")
            return
        
        try:
            # Terapkan pipeline pra-pemrosesan
            processed_image = self.preprocessor.preprocess_pipeline()
            
            if processed_image is not None:
                self.display_image(processed_image, self.display_preprocessing)
                print("Pra-pemrosesan berhasil diselesaikan")
            else:
                QMessageBox.warning(self, "Kesalahan", "Pra-pemrosesan gagal!")
                
        except Exception as e:
            QMessageBox.critical(self, "Kesalahan", f"Error pra-pemrosesan: {str(e)}")
    def apply_morphological_operations(self):
        """Terapkan thresholding Otsu dan morphological opening."""
        if self.current_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan muat gambar terlebih dahulu!")
            return
        
        try:
            # Terapkan thresholding Otsu
            binary_image = self.preprocessor.otsu_threshold()
            
            # Terapkan morphological opening
            final_image = self.preprocessor.morphological_opening()
            if final_image is not None:
                self.display_image(final_image, self.display_ekstraksi)
                print("Operasi morfologi berhasil diselesaikan")
            else:
                QMessageBox.warning(self, "Kesalahan", "Operasi morfologi gagal!")
                
        except Exception as e:
            QMessageBox.critical(self, "Kesalahan", f"Error operasi morfologi: {str(e)}")
    
    def perform_ocr(self):
        """Lakukan OCR pada gambar yang telah diproses."""
        processed_image = self.preprocessor.get_processed_image()
        if processed_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan proses gambar terlebih dahulu!")
            return
        
        try:
            # Lakukan OCR
            result = self.ocr_processor.process_image(processed_image, draw_boxes=True)
            
            # Tampilkan hasil
            extracted_text = result['text']
            
            if 'image_with_boxes' in result:
                self.display_image(result['image_with_boxes'], self.display_ekstraksi)
            
            # Perbarui text browser
            if hasattr(self, 'textBrowser'):
                self.textBrowser.setText(extracted_text)
            print("Hasil OCR:")
            print("=" * 50)
            print(f"Teks yang diekstrak: {extracted_text}")
            print(f"Jumlah karakter yang terdeteksi: {len(result['boxes'])}")
            print("=" * 50)        
        except Exception as e:
            QMessageBox.critical(self, "Kesalahan", f"Error OCR: {str(e)}")

    def load_ml_model(self):
        """Memuat dan menggunakan model pengenalan alfabet untuk deteksi karakter dengan pra-pemrosesan konsisten pelatihan."""
        original_image = self.preprocessor.get_original_image()
        processed_image = self.preprocessor.get_processed_image()
        
        if original_image is None:
            QtWidgets.QMessageBox.warning(self, "Peringatan", "Belum ada gambar yang dimuat untuk pengenalan alfabet")
            return
        
        try:
            # Impor modul yang dibutuhkan
            import cv2
            import numpy as np
            import traceback
            from alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            print("Memulai pengenalan karakter alfabet...")
            print("=" * 60)
            
            # Inisialisasi pengenal alfabet
            recognizer = AlphabeticRecognizer()
            if not recognizer.load_model():
                QtWidgets.QMessageBox.critical(self, "Kesalahan", "Gagal memuat model pengenalan alfabet")
                return
            
            print("✓ Model pengenalan alfabet berhasil dimuat")
            
            # Gunakan gambar yang telah diproses jika tersedia, jika tidak gunakan yang asli
            working_image = processed_image if processed_image is not None else original_image
            
            # Terapkan pipeline pra-pemrosesan konsisten pelatihan untuk akurasi lebih baik
            print("Menerapkan pra-pemrosesan konsisten pelatihan...")
            preprocessed_for_segmentation = self._apply_training_preprocessing(working_image)
            
            # Lakukan segmentasi karakter untuk menemukan bounding box karakter
            print("Melakukan segmentasi karakter...")
            bboxes = self._segment_characters(preprocessed_for_segmentation)
            
            if not bboxes:
                print("Segmentasi utama gagal, mencoba metode cadangan...")
                # Coba segmentasi cadangan pada gambar asli
                bboxes = self._segment_characters_fallback(original_image)
            
            print(f"Ditemukan {len(bboxes)} area karakter untuk pengenalan")
            
            # Prediksi karakter menggunakan pengenal alfabet
            detections = []
            if bboxes:
                print("Melakukan pengenalan karakter...")
                detections = recognizer.predict_characters_from_image(original_image, bboxes)
                print(f"Pengenalan selesai untuk {len(detections)} karakter")
            
            # Buat visualisasi komprehensif
            result_image = self._create_recognition_visualization(original_image, detections)
            
            # Tampilkan hasil
            if detections:
                # Tampilkan hasil di jendela terpisah
                cv2.imshow("Hasil Pengenalan Alfabet", result_image)
                cv2.waitKey(1)
            
                
                # Perbarui text browser jika tersedia
                recognized_text = ""
                for detection in detections:
                    char = detection.get('character', '?')
                    if char != '?' and char.strip():
                        recognized_text += char
                
                if hasattr(self, 'textBrowser') and recognized_text:
                    self.textBrowser.setText(recognized_text)
                
                # Log hasil detail
                print("=" * 60)
                print("HASIL PENGENALAN ALFABET")
                print("=" * 60)
                
                # Hitung tingkat kepercayaan
                high_conf_count = 0
                medium_conf_count = 0
                low_conf_count = 0
                
                for i, detection in enumerate(detections):
                    char = detection.get('character', '?')
                    conf = detection.get('confidence', 0.0)
                    status = detection.get('status', 'unknown')
                    bbox = detection.get('bbox', None)
                    
                    # Hitung tingkat kepercayaan
                    if status == 'high_confidence':
                        high_conf_count += 1
                    elif status == 'success':
                        medium_conf_count += 1
                    else:
                        low_conf_count += 1
                    
                    bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})" if bbox else "N/A"
                    print(f"[{i+1:2d}] Karakter: '{char}' | Kepercayaan: {conf:.3f} | Status: {status} | BBox: {bbox_str}")
                
                print("-" * 60)
                print(f"Teks yang dikenali: '{recognized_text}'")
                print(f"Total Karakter Terdeteksi: {len(detections)}")
                print(f"Karakter Valid: {len([d for d in detections if d.get('character', '?') != '?'])}")
                print(f"Kepercayaan Tinggi: {high_conf_count} | Kepercayaan Sedang: {medium_conf_count} | Kepercayaan Rendah: {low_conf_count}")
                print("=" * 60)
                
                # Tampilkan pesan sukses
                QtWidgets.QMessageBox.information(
                    self, 
                    "Pengenalan Selesai", 
                    f"Berhasil mengenali {len(detections)} karakter.\nTeks yang dikenali: '{recognized_text}'"
                )
                
            else:
                # Tidak ada karakter terdeteksi - buat gambar umpan balik informatif
                blank_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
                
                # Tambahkan beberapa baris umpan balik
                cv2.putText(blank_image, "Tidak ada karakter terdeteksi", 
                           (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(blank_image, "Saran:", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)
                cv2.putText(blank_image, "1. Periksa kualitas dan kontras gambar", 
                           (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)
                cv2.putText(blank_image, "2. Coba atur parameter pra-pemrosesan", 
                           (70, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)
                cv2.putText(blank_image, "3. Pastikan karakter terlihat jelas", 
                           (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)
                cv2.putText(blank_image, "4. Periksa orientasi teks yang benar", 
                           (70, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)
                
                # Tampilkan gambar umpan balik
                cv2.imshow("Hasil Pengenalan Alfabet", blank_image)
                cv2.waitKey(1)
                
                # Tampilkan di GUI
                self.display_image(blank_image, self.display_ekstraksi)
                
                # Log informasi diagnostik
                print("=" * 60)
                print("PENGENALAN ALFABET: TIDAK ADA KARAKTER TERDETEKSI")
                print("=" * 60)
                print("Informasi Diagnostik:")
                print(f"- Bentuk gambar asli: {original_image.shape}")
                print(f"- Gambar hasil proses tersedia: {processed_image is not None}")
                if processed_image is not None:
                    print(f"- Bentuk gambar hasil proses: {processed_image.shape}")
                print(f"- Bentuk gambar kerja: {working_image.shape}")
                print(f"- Segmentasi menemukan: {len(bboxes)} area")
                print("=" * 60)
                
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Tidak Ada Karakter Terdeteksi", 
                    "Tidak ada karakter yang terdeteksi pada gambar.\n\nSaran:\n"
                    "• Periksa kualitas dan kontras gambar\n"
                    "• Coba atur parameter pra-pemrosesan\n"
                    "• Pastikan karakter terlihat jelas\n"
                    "• Periksa orientasi teks yang benar"
                )
                
        except ImportError as e:
            error_msg = f"Gagal mengimpor modul pengenalan alfabet: {str(e)}"
            QtWidgets.QMessageBox.critical(self, "Kesalahan Impor", error_msg)
            print(f"Kesalahan Impor: {error_msg}")
            
        except Exception as e:
            error_msg = f"Pengenalan alfabet gagal: {str(e)}"
            QtWidgets.QMessageBox.critical(self, "Kesalahan", error_msg)
            print(f"Kesalahan pada pengenalan alfabet: {str(e)}")            
            import traceback
            traceback.print_exc()    
    def _apply_training_preprocessing(self, image):
        """Terapkan pipeline pra-pemrosesan yang sama seperti saat pelatihan untuk konsistensi."""
        try:
            import cv2
            import numpy as np
            import cv2
            import numpy as np
            
            print("Menerapkan pipeline pra-pemrosesan konsisten pelatihan...")
            
            # Konversi ke grayscale jika perlu
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                print("✓ Dikonversi ke grayscale")
            else:
                gray = image.copy()
            
            # Langkah 1: Gaussian blur untuk reduksi noise (sama seperti pelatihan)
            blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
            print("✓ Gaussian blur diterapkan")
            
            # Langkah 2: Median filter untuk penghilangan noise tambahan (sama seperti pelatihan)
            denoised = cv2.medianBlur(blurred, 3)
            print("✓ Median filter diterapkan")
            
            # Langkah 3: Thresholding Otsu (sama seperti pelatihan)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"✓ Thresholding Otsu diterapkan (threshold: {_})")
            
            # Langkah 4: Operasi morfologi (sama seperti pelatihan)
            # Opening untuk menghilangkan noise kecil
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            print("✓ Morphological opening diterapkan")
            
            # Closing untuk menghubungkan komponen yang berdekatan
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
            print("✓ Morphological closing diterapkan")
            
            # Langkah 5: Pembersihan akhir dengan erosi-dilasi (konsistensi pelatihan)
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            cleaned = cv2.erode(cleaned, kernel_clean, iterations=1)
            cleaned = cv2.dilate(cleaned, kernel_clean, iterations=1)
            print("✓ Pembersihan akhir diterapkan")
            
            print("✓ Pra-pemrosesan konsisten pelatihan selesai")
            return cleaned
            
        except Exception as e:
            print(f"Kesalahan pada pra-pemrosesan pelatihan: {e}")
            # Kembalikan threshold biner sederhana sebagai cadangan
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary    
    def _segment_characters(self, binary_image):
        """Segmentasi karakter individual dari gambar yang telah dipra-pemrosesan."""
        try:
            import cv2
            import cv2
            
            # Temukan kontur
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bboxes = []
            for contour in contours:
                # Dapatkan bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter berdasarkan ukuran dan rasio aspek (mirip konfigurasi pelatihan)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Terapkan kriteria filter
                if (area >= 50 and  # MIN_CONTOUR_AREA
                    8 <= w <= 200 and  # MIN/MAX_BBOX_WIDTH  
                    12 <= h <= 200 and  # MIN/MAX_BBOX_HEIGHT
                    0.1 <= aspect_ratio <= 4.0):  # MIN/MAX_ASPECT_RATIO
                    
                    bboxes.append((x, y, w, h))
            
            # Urutkan bboxes dari kiri ke kanan untuk urutan baca yang benar
            bboxes.sort(key=lambda bbox: bbox[0])
            
            print(f"Ditemukan {len(bboxes)} kandidat karakter setelah filter")
            return bboxes
            
        except Exception as e:
            print(f"Kesalahan pada segmentasi karakter: {e}")
            return []

    def _segment_characters_fallback(self, image):
        """Segmentasi karakter cadangan menggunakan bounding box OCR."""
        try:
            from image_processing.ocr import OCRProcessor
            
            ocr_processor = OCRProcessor()
            boxes = ocr_processor.get_bounding_boxes(image)
            
            # Konversi box OCR ke format standar (x, y, w, h)
            bboxes = []
            h_img = image.shape[0]
            
            for box in boxes:
                x = box['x']
                y = h_img - box['h']  # Konversi sistem koordinat OCR
                w = box['w'] - box['x']
                h = box['h'] - box['y']
                
                if w > 0 and h > 0:
                    bboxes.append((x, y, w, h))
            
            print(f"Segmentasi cadangan menemukan {len(bboxes)} karakter")
            return bboxes
            
        except Exception as e:
            print(f"Kesalahan pada segmentasi cadangan: {e}")
            return []    
    def _create_recognition_visualization(self, image, detections):
        """Buat visualisasi dengan bounding box dan hasil pengenalan."""
        try:
            import cv2
            import numpy as np
            import cv2
            import numpy as np
            
            # Buat salinan gambar asli untuk visualisasi
            result_image = image.copy()
            
            for detection in detections:
                bbox = detection['bbox']
                char = detection['character']
                confidence = detection['confidence']
                status = detection['status']
                
                x, y, w, h = bbox
                
                # Pilih warna berdasarkan kepercayaan dan status
                if status == 'high_confidence':
                    color = (0, 255, 0)  # Hijau untuk kepercayaan tinggi
                elif status == 'success':
                    color = (0, 165, 255)  # Oranye untuk kepercayaan sedang
                else:
                    color = (0, 0, 255)  # Merah untuk kepercayaan rendah
                
                # Gambar bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Siapkan teks label
                label = f"{char} ({confidence:.2f})"
                
                # Hitung ukuran teks untuk latar belakang
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Gambar latar belakang untuk teks
                cv2.rectangle(result_image, 
                             (x, y - text_height - baseline - 5), 
                             (x + text_width + 5, y), 
                             color, -1)
                
                # Gambar teks
                cv2.putText(result_image, label, 
                           (x + 2, y - baseline - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return result_image
            
        except Exception as e:
            print(f"Kesalahan membuat visualisasi: {e}")
            return image
    
    def display_image(self, image, display_widget):
        """
        Tampilkan gambar pada widget yang ditentukan.
        
        Argumen:
            image: Gambar yang akan ditampilkan (numpy array)
            display_widget: Widget QLabel untuk menampilkan gambar
        """
        try:
            # Tentukan format gambar
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    q_format = QImage.Format_RGBA8888
                else:
                    q_format = QImage.Format_RGB888
            else:
                q_format = QImage.Format_Indexed8
            
            # Konversi ke QImage
            h, w = image.shape[:2]
            bytes_per_line = image.strides[0] if len(image.shape) > 1 else w
            
            q_image = QImage(image.data, w, h, bytes_per_line, q_format)
            
            # Konversi ruang warna jika diperlukan
            if len(image.shape) == 3:
                q_image = q_image.rgbSwapped()
            
            # Tampilkan pada widget
            pixmap = QPixmap.fromImage(q_image)
            display_widget.setPixmap(pixmap)
            display_widget.setAlignment(QtCore.Qt.AlignCenter)
            display_widget.setScaledContents(True)
        except Exception as e:
            print(f"Kesalahan menampilkan gambar: {e}")

    def closeEvent(self, event):
        """Menangani event saat aplikasi ditutup."""        
        reply = QMessageBox.question(
            self, 
            'Keluar Aplikasi', 
            'Apakah Anda yakin ingin keluar?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
    # ================================
    # FUNGSI ML PROCESSING TERPISAH
    # ================================
    
    def apply_ml_preprocessing(self):
        """TAHAP 1 ML: Terapkan pra-pemrosesan khusus untuk model ML."""
        original_image = self.preprocessor.get_original_image()
        processed_image = self.preprocessor.get_processed_image()
        
        if original_image is None:
            QMessageBox.warning(self, "Peringatan", "Belum ada gambar yang dimuat")
            return
        
        try:
            print("=== TAHAP 1 ML: PRA-PEMROSESAN KHUSUS ===")
            
            # Gunakan gambar yang telah diproses jika tersedia, jika tidak gunakan yang asli
            working_image = processed_image if processed_image is not None else original_image
            
            # Terapkan pra-pemrosesan konsisten pelatihan
            self.ml_preprocessed_image = self._apply_training_preprocessing(working_image)
            
            if self.ml_preprocessed_image is not None:
                # Tampilkan hasil pra-pemrosesan ML
                self.display_image(self.ml_preprocessed_image, self.display_preprocessing)
                
                print("✓ Pra-pemrosesan ML berhasil")
                QMessageBox.information(self, "Tahap 1 Selesai", 
                                      "Pra-pemrosesan ML berhasil!\nLanjutkan ke Tahap 2: Segmentasi Karakter")
            else:
                QMessageBox.warning(self, "Kesalahan", "Pra-pemrosesan ML gagal!")
                
        except Exception as e:
            QMessageBox.critical(self, "Kesalahan", f"Error pra-pemrosesan ML: {str(e)}")
            print(f"Error pada pra-pemrosesan ML: {str(e)}")
    
    def perform_character_segmentation(self):
        """TAHAP 2 ML: Lakukan segmentasi karakter untuk menemukan bounding boxes."""
        if self.ml_preprocessed_image is None:
            QMessageBox.warning(self, "Peringatan", 
                              "Silakan lakukan pra-pemrosesan ML terlebih dahulu (Tahap 1)!")
            return
        
        try:
            print("=== TAHAP 2 ML: SEGMENTASI KARAKTER ===")
            
            # Lakukan segmentasi karakter
            self.character_bboxes = self._segment_characters(self.ml_preprocessed_image)
            
            if not self.character_bboxes:
                print("Segmentasi utama gagal, mencoba metode cadangan...")
                # Coba segmentasi cadangan pada gambar asli
                original_image = self.preprocessor.get_original_image()
                self.character_bboxes = self._segment_characters_fallback(original_image)
            
            print(f"Ditemukan {len(self.character_bboxes)} area karakter")
            
            if self.character_bboxes:
                # Buat visualisasi dengan bounding boxes
                segmentation_image = self._create_segmentation_visualization()
                
                # Tampilkan hasil segmentasi
                self.display_image(segmentation_image, self.display_ekstraksi)
                
                print("✓ Segmentasi karakter berhasil")
                QMessageBox.information(self, "Tahap 2 Selesai", 
                                      f"Segmentasi berhasil! Ditemukan {len(self.character_bboxes)} karakter.\n"
                                      f"Lanjutkan ke Tahap 3: Muat Model ML")
            else:
                QMessageBox.warning(self, "Peringatan", 
                                  "Tidak ada karakter yang terdeteksi!\n"
                                  "Coba periksa kualitas gambar atau parameter pra-pemrosesan.")
                
        except Exception as e:
            QMessageBox.critical(self, "Kesalahan", f"Error segmentasi karakter: {str(e)}")
            print(f"Error pada segmentasi karakter: {str(e)}")
    
    def load_and_prepare_ml_model(self):
        """TAHAP 3 ML: Muat dan persiapkan model ML untuk pengenalan."""
        try:
            print("=== TAHAP 3 ML: MEMUAT MODEL ML ===")
            
            # Impor modul yang dibutuhkan
            from alphabetic_recognition.recognizer import AlphabeticRecognizer
            
            # Inisialisasi dan muat model
            recognizer = AlphabeticRecognizer()
            if recognizer.load_model():
                self.ml_model = recognizer
                print("✓ Model ML berhasil dimuat")
                
                QMessageBox.information(self, "Tahap 3 Selesai", 
                                      "Model ML berhasil dimuat!\n"
                                      "Lanjutkan ke Tahap 4: Pengenalan Karakter")
            else:
                self.ml_model = None
                QMessageBox.critical(self, "Kesalahan", "Gagal memuat model ML!")
                
        except ImportError as e:
            error_msg = f"Gagal mengimpor modul pengenalan alfabet: {str(e)}"
            QMessageBox.critical(self, "Kesalahan Impor", error_msg)
            print(f"Kesalahan Impor: {error_msg}")
            
        except Exception as e:
            error_msg = f"Gagal memuat model ML: {str(e)}"
            QMessageBox.critical(self, "Kesalahan", error_msg)
            print(f"Error memuat model ML: {str(e)}")
    
    def perform_ml_character_recognition(self):
        """TAHAP 4 ML: Lakukan pengenalan karakter menggunakan model ML."""
        # Cek apakah semua tahap sebelumnya sudah selesai
        if self.ml_preprocessed_image is None:
            QMessageBox.warning(self, "Peringatan", 
                              "Silakan lakukan pra-pemrosesan ML terlebih dahulu (Tahap 1)!")
            return
        
        if not self.character_bboxes:
            QMessageBox.warning(self, "Peringatan", 
                              "Silakan lakukan segmentasi karakter terlebih dahulu (Tahap 2)!")
            return
        
        if self.ml_model is None:
            QMessageBox.warning(self, "Peringatan", 
                              "Silakan muat model ML terlebih dahulu (Tahap 3)!")
            return
        
        try:
            print("=== TAHAP 4 ML: PENGENALAN KARAKTER ===")
            
            original_image = self.preprocessor.get_original_image()
            
            # Lakukan prediksi karakter
            print("Melakukan pengenalan karakter...")
            self.ml_detections = self.ml_model.predict_characters_from_image(
                original_image, self.character_bboxes)
            print(f"Pengenalan selesai untuk {len(self.ml_detections)} karakter")
            
            # Buat visualisasi hasil akhir
            result_image = self._create_recognition_visualization(original_image, self.ml_detections)
            
            # Tampilkan hasil
            if self.ml_detections:
                # Tampilkan hasil di GUI
                self.display_image(result_image, self.display_ekstraksi)
                
                # Ekstrak teks yang dikenali
                recognized_text = ""
                for detection in self.ml_detections:
                    char = detection.get('character', '?')
                    if char != '?' and char.strip():
                        recognized_text += char
                
                # Update text browser jika tersedia
                if hasattr(self, 'textBrowser') and recognized_text:
                    self.textBrowser.setText(recognized_text)
                
                # Log hasil detail
                self._log_ml_results(recognized_text)
                
                # Tampilkan pesan sukses
                QMessageBox.information(
                    self, 
                    "Pengenalan ML Selesai", 
                    f"Berhasil mengenali {len(self.ml_detections)} karakter.\n"
                    f"Teks yang dikenali: '{recognized_text}'"
                )
                
            else:
                # Tidak ada hasil
                QMessageBox.warning(self, "Peringatan", 
                                  "Tidak ada karakter yang berhasil dikenali!")
                
        except Exception as e:
            error_msg = f"Pengenalan karakter ML gagal: {str(e)}"
            QMessageBox.critical(self, "Kesalahan", error_msg)
            print(f"Error pada pengenalan karakter ML: {str(e)}")
    def _create_segmentation_visualization(self):
        """Buat visualisasi hasil segmentasi dengan bounding boxes."""
        try:
            import cv2
            import numpy as np
            
            original_image = self.preprocessor.get_original_image()
            result_image = original_image.copy()
            
            # Gambar bounding boxes untuk setiap karakter
            for i, (x, y, w, h) in enumerate(self.character_bboxes):
                # Warna biru untuk segmentasi
                color = (255, 0, 0)  # BGR format
                
                # Gambar bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Tambahkan label nomor
                label = f"{i+1}"
                cv2.putText(result_image, label, 
                           (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return result_image
            
        except Exception as e:
            print(f"Kesalahan membuat visualisasi segmentasi: {e}")
            return self.preprocessor.get_original_image()
    
    def _log_ml_results(self, recognized_text):
        """Log hasil detail ML processing."""
        print("=" * 60)
        print("HASIL PENGENALAN ML BERTAHAP")
        print("=" * 60)
        
        # Hitung tingkat kepercayaan
        high_conf_count = 0
        medium_conf_count = 0
        low_conf_count = 0
        
        for i, detection in enumerate(self.ml_detections):
            char = detection.get('character', '?')
            conf = detection.get('confidence', 0.0)
            status = detection.get('status', 'unknown')
            bbox = detection.get('bbox', None)
            
            # Hitung tingkat kepercayaan
            if status == 'high_confidence':
                high_conf_count += 1
            elif status == 'success':
                medium_conf_count += 1
            else:
                low_conf_count += 1
            
            bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})" if bbox else "N/A"
            print(f"[{i+1:2d}] Karakter: '{char}' | Kepercayaan: {conf:.3f} | Status: {status} | BBox: {bbox_str}")
        
        print("-" * 60)
        print(f"Teks yang dikenali: '{recognized_text}'")
        print(f"Total Karakter Terdeteksi: {len(self.ml_detections)}")
        print(f"Karakter Valid: {len([d for d in self.ml_detections if d.get('character', '?') != '?'])}")
        print(f"Kepercayaan Tinggi: {high_conf_count} | Kepercayaan Sedang: {medium_conf_count} | Kepercayaan Rendah: {low_conf_count}")
        print("=" * 60)

    def reset_ml_processing(self):
        """Reset semua data ML processing dan tampilan untuk memulai dari awal."""
        try:
            # Reset semua data ML processing
            self.ml_preprocessed_image = None
            self.character_bboxes = []
            self.ml_model = None
            self.ml_detections = []
            
            # Reset display widgets (sesuaikan dengan nama widget di GUI Anda)
            if hasattr(self, 'display_image'):
                self.display_citra.clear()

            if hasattr(self, 'display_preprocessing'):
                self.display_preprocessing.clear()
            
            if hasattr(self, 'display_ekstraksi'):
                self.display_ekstraksi.clear()
            
            # Reset text browser jika ada
            if hasattr(self, 'textBrowser'):
                self.textBrowser.clear()
            
            # Tutup jendela OpenCV jika masih terbuka
            try:
                import cv2
                cv2.destroyWindow("Hasil Pengenalan Alfabet")
                cv2.destroyAllWindows()
            except:
                pass
            
            QMessageBox.information(
                self, 
                "Reset Selesai", 
                "Reset ML processing berhasil!\n\n"
                "Anda dapat memulai dari Tahap 1 kembali."
            )
            
        except Exception as e:
            error_msg = f"Error saat reset: {str(e)}"
            print(f"Error pada reset ML processing: {error_msg}")
            QMessageBox.warning(self, "Peringatan", f"Reset tidak sepenuhnya berhasil:\n{error_msg}")

    # ================================
    # FUNGSI HELPER YANG SUDAH ADA
    # ================================
