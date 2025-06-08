# Dokumentasi Alur Pemrosesan ML Model (.pkl) - Pengenalan Alfabet

## 1. Overview & Comparison

### Pendekatan ML Model (.pkl)
Pendekatan ini menggunakan model machine learning yang telah dilatih sebelumnya (AlphabeticRecognizer) untuk mengenali karakter alfabet (A-Z, 0-9). Model disimpan dalam format .pkl dan menggunakan ekstraksi fitur komprehensif dengan klasifikasi berbasis supervised learning.

### Keunggulan Utama
- **Akurasi Tinggi**: Model terlatih khusus untuk dataset karakter alfabet
- **Fitur Ekstraksi Komprehensif**: Kombinasi 6 jenis fitur (Hu Moments, HOG, Geometric, Projection, Zoning, Crossing)
- **Konfigurasi Terpusat**: Parameter dapat disesuaikan melalui config.py
- **Optimasi Performa**: Feature caching, multiprocessing, dan batch processing
- **Confidence Scoring**: Memberikan skor kepercayaan untuk setiap prediksi

---

## 2. Visualizable Process Stages

```
INPUT IMAGE
     ↓
[1] ML Preprocessing
     ↓
[2] Character Segmentation  
     ↓
[3] Feature Extraction
     ↓
[4] ML Classification
     ↓
OUTPUT RESULTS
```

### Stage Flow Detail:
1. **ML Preprocessing**: Normalisasi gambar khusus untuk ML model
2. **Character Segmentation**: Deteksi dan isolasi karakter individual  
3. **Feature Extraction**: Ekstraksi 6 jenis fitur dari setiap karakter
4. **ML Classification**: Prediksi menggunakan model terlatih + confidence scoring

---

## 3. Detailed Stage Breakdown

### Stage 1: ML Preprocessing
**File**: `alphabetic_recognition/recognizer.py` - `preprocess_character_image()`

**Input**: Raw image ROI (Region of Interest)
**Output**: Normalized binary image (28x28 pixels)

**Proses Detail**:
```python
def preprocess_character_image(self, image_roi, target_size=(28,28)):
    # 1. Konversi ke Grayscale
    if len(image_roi.shape) == 3:
        gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise Reduction (Gaussian + Median Blur)
    denoised = cv2.GaussianBlur(gray_image, (3,3), 1, sigmaY=1)
    denoised = cv2.medianBlur(denoised, 3)
    
    # 3. Binarization (Otsu Thresholding)
    _, binary_image = cv2.threshold(denoised, 0, 255, 
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Morphological Operations
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_3x3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_3x3)
    
    # 5. Resize to Standard Size
    normalized = cv2.resize(cleaned, target_size, interpolation=cv2.INTER_AREA)
    
    # 6. Orientation Correction
    if white_pixels > total_pixels * 0.5:
        normalized = cv2.bitwise_not(normalized)
```

**Konfigurasi** (dari `config.py`):
- `CHAR_IMAGE_SIZE = (28, 28)` - Ukuran standar karakter
- `GAUSSIAN_KERNEL_SIZE = (3, 3)` - Kernel Gaussian blur
- `MEDIAN_FILTER_SIZE = 3` - Ukuran median filter
- `THRESHOLD_TYPE = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`

### Stage 2: Character Segmentation
**File**: GUI integration - character bounding box detection

**Input**: Preprocessed binary image
**Output**: List of character bounding boxes

**Proses Detail**:
```python
def perform_character_segmentation():
    # 1. Contour Detection
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Contour Filtering
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Filter berdasarkan kriteria dari config.py
        if (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA_RATIO * image_area and
            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and
            MIN_BBOX_WIDTH <= w <= MAX_BBOX_WIDTH and
            MIN_BBOX_HEIGHT <= h <= MAX_BBOX_HEIGHT):
            valid_contours.append(contour)
    
    # 3. Sorting Left-to-Right
    bboxes = [cv2.boundingRect(c) for c in valid_contours]
    bboxes.sort(key=lambda box: box[0])  # Sort by x-coordinate
```

**Parameter Filter** (dari `config.py`):
- `MIN_CONTOUR_AREA = 30` - Area minimum kontur
- `MAX_CONTOUR_AREA_RATIO = 0.4` - Rasio maksimum area
- `MIN_ASPECT_RATIO = 0.1` - Rasio aspek minimum
- `MAX_ASPECT_RATIO = 4.0` - Rasio aspek maksimum
- `MIN_BBOX_WIDTH = 8`, `MIN_BBOX_HEIGHT = 12` - Dimensi minimum bounding box

### Stage 3: Feature Extraction
**File**: `alphabetic_recognition/recognizer.py` - `extract_features_from_character()`

**Input**: Binary character image (28x28)
**Output**: Weighted feature vector (72 dimensions)

**6 Jenis Fitur yang Diekstrak**:

#### 3.1 Hu Moments (7 features)
```python
moments = cv2.moments(binary_char_image)
if moments['m00'] != 0:
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log normalization untuk stabilitas
    hu_moments = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
features.extend(hu_moments * FEATURE_WEIGHTS['hu_moments'])  # weight: 1.0
```

#### 3.2 HOG Features (36 features)
```python
# Histogram of Oriented Gradients
hog_descriptor = cv2.HOGDescriptor(
    _winSize=(28,28), _blockSize=(14,14), _blockStride=(7,7),
    _cellSize=(7,7), _nbins=9, _derivAperture=1
)
hog_features = hog_descriptor.compute(resized_img)
features.extend(hog_features.flatten() * FEATURE_WEIGHTS['hog'])  # weight: 2.0
```

#### 3.3 Geometric Features (5 features)
```python
max_contour = max(contours, key=cv2.contourArea)
area = cv2.contourArea(max_contour)
perimeter = cv2.arcLength(max_contour, True)
x, y, w, h = cv2.boundingRect(max_contour)

aspect_ratio = float(w) / h
area_ratio = area / (w * h)
perimeter_area_ratio = perimeter / area if perimeter > 0 else 0

hull = cv2.convexHull(max_contour)
hull_area = cv2.contourArea(hull)
solidity = area / hull_area if hull_area > 0 else 0
extent = area / (w * h)

geometric_features = [aspect_ratio, area_ratio, perimeter_area_ratio, 
                     solidity, extent]
features.extend(geometric_features * FEATURE_WEIGHTS['geometric'])  # weight: 1.5
```

#### 3.4 Projection Features (6 features)
```python
# Horizontal and Vertical Projections
h_proj = np.sum(binary_char_image, axis=1)  # Sum along rows
v_proj = np.sum(binary_char_image, axis=0)  # Sum along columns

h_mean, h_std, h_max = np.mean(h_proj), np.std(h_proj), np.max(h_proj)
v_mean, v_std, v_max = np.mean(v_proj), np.std(v_proj), np.max(v_proj)

projection_features = [h_mean, h_std, h_max, v_mean, v_std, v_max]
features.extend(projection_features * FEATURE_WEIGHTS['projection'])  # weight: 1.2
```

#### 3.5 Zoning Features (16 features)
```python
# Divide image into 4x4 grid and calculate density
height, width = binary_char_image.shape
zone_h, zone_w = height // 4, width // 4
zoning_features = []

for i in range(4):
    for j in range(4):
        zone = binary_char_image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
        density = np.sum(zone) / (255 * zone.size) if zone.size > 0 else 0
        zoning_features.append(density)

features.extend(zoning_features * FEATURE_WEIGHTS['zoning'])  # weight: 1.3
```

#### 3.6 Crossing Features (2 features)
```python
# Count transitions in middle row and column
mid_row = height // 2
mid_col = width // 2

# Horizontal crossings
h_crossings = 0
row = binary_char_image[mid_row, :]
for k in range(1, len(row)):
    if row[k] != row[k-1]:
        h_crossings += 1

# Vertical crossings  
v_crossings = 0
col = binary_char_image[:, mid_col]
for k in range(1, len(col)):
    if col[k] != col[k-1]:
        v_crossings += 1

crossing_features = [h_crossings, v_crossings]
features.extend(crossing_features * FEATURE_WEIGHTS['crossing'])  # weight: 0.8
```

**Feature Weights** (dari `config.py`):
```python
FEATURE_WEIGHTS = {
    'hu_moments': 1.0,    # Shape descriptors
    'hog': 2.0,           # Most important for character recognition  
    'geometric': 1.5,     # Important structural features
    'projection': 1.2,    # Useful for character discrimination
    'zoning': 1.3,        # Spatial density information
    'crossing': 0.8       # Less critical but still useful
}
```

### Stage 4: ML Classification
**File**: `alphabetic_recognition/recognizer.py` - `predict_character()`

**Input**: Feature vector (72 dimensions)
**Output**: Character prediction + confidence score

**Proses Detail**:
```python
def predict_character(self, image_roi, return_probabilities=False):
    # 1. Preprocessing
    preprocessed = self.preprocess_character_image(image_roi)
    
    # 2. Feature Extraction
    features = self.extract_features_from_character(preprocessed)
    
    # 3. Feature Size Validation
    if features.size != self.feature_size:
        # Resize atau padding jika diperlukan
        features = self._normalize_feature_size(features)
    
    # 4. Model Prediction
    prediction = self.classifier.predict([features])[0]
    
    # 5. Confidence Calculation
    confidence = self._calculate_confidence(features)
    
    # 6. Threshold Application
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        prediction = "?"
        confidence = 0.0
    
    return str(prediction), float(confidence)
```

**Confidence Calculation**:
```python
def _calculate_confidence(self, features):
    if hasattr(self.classifier, 'predict_proba'):
        probabilities = self.classifier.predict_proba([features])[0]
        confidence = np.max(probabilities)
    elif hasattr(self.classifier, 'decision_function'):
        decision_scores = self.classifier.decision_function([features])
        confidence = self._normalize_decision_score(decision_scores)
    else:
        # Fallback: feature-based confidence
        confidence = self._calculate_feature_based_confidence(features)
    
    return confidence
```

**Threshold Configuration**:
- `MIN_CONFIDENCE_THRESHOLD = 0.3` - Minimum confidence untuk prediksi valid
- `HIGH_CONFIDENCE_THRESHOLD = 0.8` - Threshold untuk high-confidence predictions

---

## 4. Advantages & Disadvantages

### ✅ Advantages

#### 4.1 Accuracy & Performance
- **High Recognition Accuracy**: Model terlatih khusus menghasilkan akurasi tinggi untuk karakter alfabet
- **Comprehensive Feature Extraction**: 6 jenis fitur memberikan representasi karakter yang kuat
- **Confidence Scoring**: Memberikan measurement kepercayaan prediksi
- **Optimized Processing**: Feature caching dan multiprocessing untuk performa optimal

#### 4.2 Flexibility & Customization  
- **Centralized Configuration**: Semua parameter dapat disesuaikan via `config.py`
- **Weighted Feature Ensemble**: Bobot fitur dapat disesuaikan berdasarkan pentingnya
- **Adaptive Preprocessing**: Fallback mechanisms untuk robustness
- **Batch Processing**: Support untuk pemrosesan banyak karakter sekaligus

#### 4.3 Technical Robustness
- **Error Handling**: Comprehensive error handling dengan fallback options
- **Memory Optimization**: Feature caching dengan LRU cache
- **Performance Tracking**: Built-in metrics untuk monitoring performa
- **Model Validation**: Validation model structure saat loading

### ❌ Disadvantages

#### 4.1 Complexity & Requirements
- **Model Dependency**: Memerlukan model .pkl yang sudah dilatih
- **Complex Pipeline**: Multi-stage processing lebih kompleks dibanding OCR langsung
- **Resource Intensive**: Feature extraction memerlukan computational resources lebih tinggi
- **Setup Complexity**: Memerlukan konfigurasi parameter yang tepat

#### 4.2 Limitations
- **Fixed Character Set**: Terbatas pada karakter yang dilatih (A-Z, 0-9)
- **Preprocessing Sensitivity**: Hasil sangat bergantung pada kualitas preprocessing
- **Model Size**: Model file .pkl bisa berukuran besar
- **Training Data Dependency**: Akurasi bergantung pada kualitas dan variasi training data

#### 4.3 Maintenance & Scalability
- **Model Updates**: Memerlukan retraining untuk karakter atau font baru
- **Version Control**: Model versioning untuk track perubahan performa
- **Feature Engineering**: Perlu expertise untuk tuning feature weights
- **Computational Cost**: Lebih expensive dibanding simple OCR untuk teks sederhana

---

## 5. Performance Metrics & Monitoring

### 5.1 Built-in Metrics
```python
# Performance tracking dalam AlphabeticRecognizer
self.prediction_count = 0
self.total_confidence = 0.0
self.total_processing_time = 0.0

def get_performance_stats(self):
    avg_confidence = self.total_confidence / self.prediction_count
    return {
        "avg_confidence": round(avg_confidence, 3),
        "prediction_count": self.prediction_count,
        "avg_processing_time": self.total_processing_time / self.prediction_count
    }
```

### 5.2 Configuration Optimization
```python
# Performance optimization settings dari config.py
USE_MULTIPROCESSING = True      # Enable multiprocessing
MAX_WORKERS = 4                 # Maximum worker processes  
BATCH_SIZE = 32                 # Batch size for processing
FEATURE_CACHE_SIZE = 1000       # Maximum cached feature vectors
PRELOAD_MODEL = True            # Preload model at startup
```

---

*Dokumentasi ini menggambarkan alur pemrosesan ML Model (.pkl) untuk pengenalan alfabet dalam aplikasi HurufinAPP. Untuk perbandingan, lihat dokumentasi terpisah untuk alur Tesseract OCR.*
