# Dokumentasi Alur Pemrosesan Tesseract OCR - Pengenalan Alfabet

## 1. Overview & Comparison

### Pendekatan Tesseract OCR
Pendekatan ini menggunakan Tesseract OCR engine yang merupakan salah satu OCR open-source terbaik yang dikembangkan oleh Google. Tesseract menggunakan teknologi neural network modern dan dapat mengenali teks dalam berbagai bahasa dengan akurasi tinggi untuk dokumen yang berkualitas baik.

### Keunggulan Utama
- **Plug-and-Play**: Tidak memerlukan training model khusus
- **Multi-language Support**: Mendukung 100+ bahasa dengan language packs
- **Industrial Standard**: Widely used dan well-tested OCR engine
- **Text Layout Analysis**: Dapat menangani complex layouts dan formatting
- **Out-of-the-box Performance**: Langsung dapat digunakan tanpa setup kompleks

---

## 2. Visualizable Process Stages

```
INPUT IMAGE
     ↓
[1] Basic Preprocessing
     ↓
[2] Morphological Operations
     ↓
[3] Tesseract Processing
     ↓
[4] Text Extraction & Bounding Boxes
     ↓
OUTPUT RESULTS
```

### Stage Flow Detail:
1. **Basic Preprocessing**: Grayscale, blur, contrast enhancement
2. **Morphological Operations**: Thresholding dan cleanup operations
3. **Tesseract Processing**: OCR engine analysis dengan built-in preprocessing
4. **Text Extraction**: Character recognition + bounding box detection

---

## 3. Detailed Stage Breakdown

### Stage 1: Basic Preprocessing
**File**: `image_processing/preprocessing.py` - `preprocess_pipeline()`

**Input**: Raw image file
**Output**: Enhanced grayscale image

**Proses Detail**:
```python
def preprocess_pipeline(self):
    """
    Pipeline preprocessing lengkap untuk OCR:
    1. Konversi grayscale
    2. Median blur  
    3. Contrast stretching
    """
    try:
        # 1. Grayscale Conversion
        self.grayscale()
        
        # 2. Noise Reduction - Median Blur
        self.median_blur(kernel_size=3)
        
        # 3. Contrast Enhancement
        self.contrast_stretching()
        
        return self.processed_image
        
    except Exception as e:
        print(f"Error pada pipeline preprocessing: {e}")
        return None
```

#### 1.1 Grayscale Conversion
```python
def grayscale(self):
    """Konversi gambar ke grayscale dengan implementasi manual."""
    if len(self.processed_image.shape) == 3:
        # Manual RGB to Grayscale: 0.299*R + 0.587*G + 0.114*B
        gray = (0.299 * self.processed_image[:, :, 2] + 
                0.587 * self.processed_image[:, :, 1] + 
                0.114 * self.processed_image[:, :, 0])
        self.processed_image = gray.astype(np.uint8)
    return self.processed_image
```

#### 1.2 Median Blur (Noise Reduction)
```python
def median_blur(self, kernel_size=3):
    """Terapkan filter median blur untuk mengurangi noise."""
    img = self.processed_image
    h, w = img.shape
    pad_size = kernel_size // 2
    output = np.zeros_like(img)
    
    # Padding dengan refleksi
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), 
                       mode='reflect')
    
    # Apply median filter
    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(window)
    
    self.processed_image = output
    return self.processed_image
```

#### 1.3 Contrast Stretching
```python
def contrast_stretching(self):
    """Terapkan contrast stretching untuk meningkatkan kontras gambar."""
    img = self.processed_image
    
    # Find min and max values
    min_val = np.min(img)
    max_val = np.max(img)
    
    # Avoid division by zero
    if max_val == min_val:
        return img
    
    # Stretch contrast to full range [0, 255]
    stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    self.processed_image = stretched
    return stretched
```

### Stage 2: Morphological Operations  
**File**: `image_processing/preprocessing.py` - morphological functions

**Input**: Enhanced grayscale image
**Output**: Binary image ready for OCR

**Proses Detail**:

#### 2.1 Otsu Thresholding
```python
def otsu_threshold(self):
    """Terapkan metode thresholding Otsu untuk binarization optimal."""
    img = self.processed_image
    
    # Calculate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    variances = np.zeros(256)
    
    # Find optimal threshold using Otsu's method
    for t in range(256):
        w1 = np.sum(hist[:t+1])      # Background weight
        w2 = np.sum(hist[t+1:])      # Foreground weight
        
        if w1 == 0 or w2 == 0:
            continue
            
        # Calculate means
        m1 = np.sum(np.arange(t + 1) * hist[:t + 1]) / w1
        m2 = np.sum(np.arange(t + 1, 256) * hist[t + 1:]) / w2
        
        # Between-class variance
        variances[t] = float(w1) * float(w2) * ((m1 - m2) ** 2)
    
    # Find optimal threshold
    threshold = np.argmax(variances)
    binary_img = self.binary(img, threshold)
    
    self.processed_image = binary_img
    return binary_img
```

#### 2.2 Morphological Opening
```python
def morphological_opening(self, kernel_size=(3, 3)):
    """
    Terapkan operasi morphological opening (erosi diikuti dilasi) 
    untuk menghilangkan noise kecil
    """
    if self.processed_image is None:
        raise ValueError("Belum ada gambar yang dimuat")
    
    # Opening = Erosion followed by Dilation
    self.morphological_erosion(kernel_size, iterations=1)
    self.morphological_dilation(kernel_size, iterations=1)
    
    return self.processed_image
```

#### 2.3 Morphological Erosion
```python
def morphological_erosion(self, kernel_size=(3, 3), iterations=1):
    """Implementasi manual morfologi erosi untuk cleanup noise."""
    img = self.processed_image
    h, w = img.shape
    kh, kw = kernel_size
    
    # Create structuring element
    strel = np.ones(kernel_size, np.uint8)
    pad_h, pad_w = kh // 2, kw // 2
    
    result = img.copy()
    
    for _ in range(iterations):
        temp = np.zeros_like(result)
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), 
                       mode='constant', constant_values=0)
        
        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                # Erosion: pixel is 1 only if all overlapping pixels are 1
                if np.all(np.logical_or(region == 255, strel == 0)):
                    temp[i, j] = 255
                else:
                    temp[i, j] = 0
        
        result = temp
    
    self.processed_image = result
    return result
```

### Stage 3: Tesseract Processing
**File**: `image_processing/ocr.py` - `OCRProcessor` class

**Input**: Preprocessed binary image
**Output**: Tesseract internal analysis

**Tesseract Setup & Configuration**:
```python
class OCRProcessor:
    def __init__(self, tesseract_path=None):
        """
        Inisialisasi pemroses OCR dengan auto-detection Tesseract path.
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            self._setup_tesseract_path()
    
    def _setup_tesseract_path(self):
        """Deteksi otomatis dan atur path Tesseract."""
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            r'C:\tools\tesseract\tesseract.exe',
            'tesseract'  # If in PATH
        ]
        
        for path in possible_paths:
            try:
                if path == 'tesseract':
                    subprocess.run([path, '--version'], 
                                 capture_output=True, check=True)
                    pytesseract.pytesseract.tesseract_cmd = path
                    return
                elif os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    return
            except Exception:
                continue
```

**Tesseract Internal Processing (Black Box)**:
1. **Page Layout Analysis**: Tesseract menganalisis struktur halaman
2. **Line Detection**: Mendeteksi baris teks
3. **Word Segmentation**: Memisahkan kata-kata
4. **Character Segmentation**: Memisahkan karakter individual
5. **Character Recognition**: Neural network recognition
6. **Linguistic Analysis**: Post-processing dengan dictionary dan language model

### Stage 4: Text Extraction & Bounding Boxes
**File**: `image_processing/ocr.py` - extraction methods

**Input**: Tesseract analysis results
**Output**: Extracted text + character coordinates

#### 4.1 Text Extraction
```python
def extract_text(self, image, config=''):
    """
    Ekstrak teks dari gambar menggunakan Tesseract OCR.
    
    Args:
        image: Gambar input (numpy array)
        config: String konfigurasi Tesseract (optional)
        
    Returns:
        str: Teks yang diekstrak
    """
    try:
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return ""
```

#### 4.2 Bounding Box Detection
```python
def get_bounding_boxes(self, image):
    """
    Dapatkan bounding box untuk karakter yang terdeteksi.
    
    Args:
        image: Gambar input (numpy array)
        
    Returns:
        list: Daftar koordinat bounding box dengan format:
              [{'char': 'A', 'x': 10, 'y': 20, 'w': 30, 'h': 40}, ...]
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
```

#### 4.3 Bounding Box Visualization
```python
def draw_bounding_boxes(self, image, boxes=None):
    """
    Gambar bounding box pada gambar untuk visualisasi.
    
    Args:
        image: Gambar input
        boxes: List bounding boxes (auto-detect jika None)
        
    Returns:
        numpy.ndarray: Gambar dengan bounding box
    """
    if boxes is None:
        boxes = self.get_bounding_boxes(image)
    
    result_image = image.copy()
    h, w = image.shape[:2]
    
    for box in boxes:
        x, y, w_box, h_box = box['x'], box['y'], box['w'], box['h']
        # Convert Tesseract coordinates to OpenCV coordinates
        cv2.rectangle(result_image, 
                     (x, h - y), 
                     (w_box, h - h_box), 
                     (0, 255, 0), 1)
    
    return result_image
```

#### 4.4 Complete OCR Processing
```python
def process_image(self, image, draw_boxes=False):
    """
    Proses OCR lengkap: ekstrak teks dan opsional gambar bounding box.
    
    Args:
        image: Gambar input (numpy array)
        draw_boxes: Apakah menggambar bounding box pada gambar
        
    Returns:
        dict: Dictionary berisi:
              - 'text': Teks yang diekstrak
              - 'boxes': List bounding boxes
              - 'image_with_boxes': Gambar dengan bounding box (jika draw_boxes=True)
    """
    result = {}
    
    # Extract text
    result['text'] = self.extract_text(image)
    
    # Get bounding boxes
    result['boxes'] = self.get_bounding_boxes(image)
    
    # Draw bounding boxes if requested
    if draw_boxes:
        result['image_with_boxes'] = self.draw_bounding_boxes(image, result['boxes'])
    
    return result
```

---

## 4. Advantages & Disadvantages

### ✅ Advantages

#### 4.1 Ease of Use & Setup
- **Zero Training Required**: Tidak perlu training model atau dataset
- **Plug-and-Play**: Install Tesseract dan langsung bisa digunakan
- **Automatic Configuration**: Auto-detection Tesseract installation path
- **Standard OCR Solution**: Industry-proven OCR engine

#### 4.2 Versatility & Capability
- **Multi-Language Support**: 100+ bahasa dengan language packs
- **Complex Text Layouts**: Handle complex document layouts
- **Various Text Orientations**: Dapat menangani teks rotasi
- **Font Flexibility**: Bekerja dengan berbagai jenis font

#### 4.3 Robust Performance
- **Google-Backed**: Developed dan maintained oleh Google
- **Continuous Updates**: Regular improvements dan bug fixes
- **Neural Network Based**: Modern LSTM neural networks
- **Production Ready**: Widely used dalam production environments

#### 4.4 Integration & Compatibility
- **Cross-Platform**: Windows, Linux, macOS support
- **Language Bindings**: Python, Java, C++, dan banyak bahasa lain
- **Open Source**: Free dan open source
- **API Flexibility**: Multiple output formats (text, hOCR, PDF, etc.)

### ❌ Disadvantages

#### 4.1 Performance Limitations
- **Quality Dependent**: Sangat bergantung pada kualitas input image
- **Preprocessing Sensitive**: Butuh preprocessing yang baik untuk hasil optimal
- **No Confidence Scores**: Tidak memberikan confidence scoring built-in
- **Processing Speed**: Bisa lebih lambat dibanding specialized models

#### 4.2 Accuracy Constraints
- **General Purpose**: Tidak dioptimasi untuk domain/font spesifik
- **Noise Sensitivity**: Sensitive terhadap noise dan artifacts
- **Character Confusion**: Sering confuse similar characters (0/O, 1/I/l)
- **Handwritten Text**: Performa buruk untuk handwritten text

#### 4.3 Configuration & Customization
- **Limited Customization**: Parameter tuning terbatas
- **Language Pack Dependency**: Perlu install language packs untuk bahasa non-English
- **No Domain Adaptation**: Tidak bisa fine-tune untuk domain spesifik
- **Black Box**: Internal processing tidak bisa dimodifikasi

#### 4.4 Technical Limitations
- **Memory Usage**: Bisa memory-intensive untuk gambar besar
- **Installation Complexity**: Tesseract installation bisa tricky di beberapa sistem
- **Version Compatibility**: Different versions bisa memberikan hasil berbeda
- **Error Handling**: Limited error information untuk debugging

---

## 5. Tesseract Configuration & Optimization

### 5.1 Tesseract Config Options
```python
# Common Tesseract configuration strings
configs = {
    'default': '',
    'digits_only': '--psm 8 -c tessedit_char_whitelist=0123456789',
    'alpha_only': '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'single_word': '--psm 8',
    'single_char': '--psm 10',
    'sparse_text': '--psm 11',
    'no_dict': '--psm 8 -c load_system_dawg=false -c load_freq_dawg=false',
}

# Usage
text = ocr.extract_text(image, config=configs['alpha_only'])
```

### 5.2 Page Segmentation Modes (PSM)
```python
PSM_MODES = {
    0: 'Orientation and script detection (OSD) only',
    1: 'Automatic page segmentation with OSD',
    2: 'Automatic page segmentation, but no OSD, or OCR',
    3: 'Fully automatic page segmentation, but no OSD (Default)',
    4: 'Assume a single column of text of variable sizes',
    5: 'Assume a single uniform block of vertically aligned text',
    6: 'Assume a single uniform block of text',
    7: 'Treat the image as a single text line',
    8: 'Treat the image as a single word',
    9: 'Treat the image as a single word in a circle',
    10: 'Treat the image as a single character',
    11: 'Sparse text. Find as much text as possible in no particular order',
    12: 'Sparse text with OSD',
    13: 'Raw line. Treat the image as a single text line'
}
```

### 5.3 Performance Optimization Tips
1. **Image Quality**: Ensure high contrast, clean images
2. **Appropriate PSM**: Choose correct page segmentation mode
3. **Character Whitelist**: Limit character set jika memungkinkan
4. **Preprocessing**: Proper image preprocessing sangat penting
5. **Language Selection**: Pilih language pack yang tepat
6. **Resolution**: 300 DPI optimal untuk most text

---

## 6. Error Handling & Troubleshooting

### 6.1 Common Issues & Solutions
```python
def robust_ocr_processing(self, image):
    """Enhanced OCR processing dengan error handling."""
    try:
        # Basic processing
        result = self.process_image(image)
        
        if not result['text'].strip():
            # Try different PSM modes jika tidak ada hasil
            for psm in [8, 7, 6, 11]:
                config = f'--psm {psm}'
                text = self.extract_text(image, config)
                if text.strip():
                    result['text'] = text
                    break
        
        return result
        
    except Exception as e:
        print(f"OCR processing error: {e}")
        return {'text': '', 'boxes': [], 'error': str(e)}
```

### 6.2 Installation Verification
```python
def verify_tesseract_installation(self):
    """Verify Tesseract installation dan capabilities."""
    try:
        # Test basic functionality
        test_image = np.ones((50, 200), dtype=np.uint8) * 255
        cv2.putText(test_image, 'TEST', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        
        text = pytesseract.image_to_string(test_image)
        
        if 'TEST' in text.upper():
            print("✓ Tesseract installation verified")
            return True
        else:
            print("❌ Tesseract not working properly")
            return False
            
    except Exception as e:
        print(f"❌ Tesseract verification failed: {e}")
        return False
```

---

*Dokumentasi ini menggambarkan alur pemrosesan Tesseract OCR untuk pengenalan alfabet dalam aplikasi HurufinAPP. Untuk perbandingan, lihat dokumentasi terpisah untuk alur ML Model (.pkl).*
