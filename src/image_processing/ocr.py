"""
Modul Optical Character Recognition untuk aplikasi Hurufin.
Menangani ekstraksi teks dan deteksi bounding box menggunakan Tesseract OCR.
"""

import cv2
import pytesseract
import os


class OCRProcessor:
    """
    Kelas untuk menangani operasi OCR menggunakan Tesseract.
    """
    
    def __init__(self, tesseract_path=None):
        """
        Inisialisasi pemroses OCR.
        
        Argumen:
            tesseract_path (str): Path ke executable tesseract. 
                                Jika None, menggunakan deteksi otomatis.
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Deteksi otomatis path Tesseract
            self._setup_tesseract_path()
    def _setup_tesseract_path(self):
        """Deteksi otomatis dan atur path Tesseract."""
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.environ.get('USERNAME', '')),
            r'C:\tools\tesseract\tesseract.exe',
            'tesseract'  # Jika ditambahkan ke PATH
        ]
        
        for path in possible_paths:
            try:
                if path == 'tesseract':
                    # Uji apakah tesseract ada di PATH
                    import subprocess
                    subprocess.run([path, '--version'], capture_output=True, check=True)
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Using Tesseract from PATH: {path}")
                    return
                elif os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Found Tesseract at: {path}")
                    return
            except Exception:
                continue
        
        # Jika tidak ada yang ditemukan, gunakan default dan beri tahu pengguna
        default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = default_path
        print(f"Warning: Tesseract not found. Using default path: {default_path}")
        print("Please install Tesseract OCR or update the path manually.")
    
    def extract_text(self, image, config=''):
        """
        Ekstrak teks dari gambar menggunakan Tesseract OCR.
        
        Argumen:
            image: Gambar input (numpy array)
            config: String konfigurasi Tesseract
            
        Return:
            str: Teks yang diekstrak
        """
        try:
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return ""
    
    def get_bounding_boxes(self, image):
        """
        Dapatkan bounding box untuk karakter yang terdeteksi.
        
        Argumen:
            image: Gambar input (numpy array)
            
        Return:
            list: Daftar koordinat bounding box
        """
        try:
            boxes = pytesseract.image_to_boxes(image)
            box_list = []
            
            for b in boxes.splitlines():
                b = b.split(' ')
                if len(b) >= 5:
                    char = b[0]
                    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                    box_list.append({
                        'char': char,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })
            
            return box_list
        except Exception as e:
            print(f"Error in bounding box detection: {e}")
            return []
    
    def draw_bounding_boxes(self, image, boxes=None):
        """
        Gambar bounding box pada gambar.
        
        Argumen:
            image: Gambar input (numpy array)
            boxes: Daftar bounding box. Jika None, deteksi otomatis.
            
        Return:
            numpy.ndarray: Gambar dengan bounding box yang digambar
        """
        if boxes is None:
            boxes = self.get_bounding_boxes(image)
        
        result_image = image.copy()
        h, w = image.shape[:2]
        
        for box in boxes:
            x, y, w_box, h_box = box['x'], box['y'], box['w'], box['h']
            # Konversi koordinat Tesseract ke koordinat OpenCV
            cv2.rectangle(result_image, 
                         (x, h - y), 
                         (w_box, h - h_box), 
                         (0, 255, 0), 1)
        
        return result_image
    
    def process_image(self, image, draw_boxes=False):
        """
        Proses OCR lengkap: ekstrak teks dan opsional gambar bounding box.
        
        Argumen:
            image: Gambar input (numpy array)
            draw_boxes: Apakah menggambar bounding box pada gambar
            
        Return:
            dict: Dictionary berisi 'text', 'boxes', dan opsional 'image_with_boxes'
        """
        result = {}
        
        # Ekstrak teks
        result['text'] = self.extract_text(image)
        
        # Dapatkan bounding boxes
        result['boxes'] = self.get_bounding_boxes(image)
        
        # Gambar bounding boxes jika diminta
        if draw_boxes:
            result['image_with_boxes'] = self.draw_bounding_boxes(image, result['boxes'])
        
        return result
