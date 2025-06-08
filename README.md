# HurufinApp - Advanced OCR Application

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-orange.svg)

**Aplikasi OCR (Optical Character Recognition) profesional dengan GUI modern untuk ekstraksi teks dari gambar**

---

## Fitur Unggulan

### Preprocessing Canggih
- **Multi-format Support**: PNG, JPG, JPEG, BMP, TIFF
- **Intelligent Preprocessing**: Grayscale, median blur, contrast stretching
- **Advanced Morphology**: Otsu thresholding, morphological operations
- **Noise Reduction**: Adaptive filtering dan edge preservation

### Dual OCR Engine
- **Tesseract OCR**: Industry-standard text recognition
- **Custom ML Model**: Advanced character segmentation dan recognition
- **Character-level Analysis**: Bounding box detection dengan confidence scoring
- **Visual Feedback**: Real-time visualization dengan color-coded results

### Modern GUI Experience
- **Professional Interface**: Clean dan intuitive PyQt5 design
- **4-Stage ML Pipeline**: Step-by-step processing workflow
- **Real-time Processing**: Live preview untuk setiap tahap
- **Export Results**: Copy text atau save processed images

---

## Arsitektur Sistem

```text
HurufinApp/
├── main.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── alphabet_recognition/      # ML training datasets
│   ├── sample_images/            # Test images
│   └── Hurufin/                  # App assets dan logos
├── models/
│   ├── alphabetic_classifier_model.pkl  # Trained ML model
│   └── feature_extractor_config.pkl     # Feature extraction config
├── src/
│   ├── gui/
│   │   ├── GUI.ui                # PyQt5 interface design
│   │   └── main_window.py        # Main application window
│   ├── image_processing/
│   │   ├── preprocessing.py      # Image preprocessing pipeline
│   │   └── ocr.py               # OCR processing engine
│   ├── alphabetic_recognition/
│   │   └── recognizer.py        # Custom ML character recognizer
│   └── utils/
│       └── helpers.py           # Utility functions
├── docs/
│   ├── user_guide.md           # User documentation
│   └── technical_docs/         # Technical documentation
└── tests/
    ├── test_ocr.py            # OCR engine tests
    └── test_preprocessing.py  # Preprocessing tests
```

## Instalasi

## Instalasi

### Prasyarat

1. **Python 3.7+**: Pastikan Python sudah terinstall di sistem Anda
2. **Tesseract OCR**: Download dan install dari [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Path instalasi default: `C:\Program Files\Tesseract-OCR\`
   - Tambahkan ke system PATH atau update path di kode

### Setup Cepat

1. **Clone repository**:

   ```bash
   git clone https://github.com/yourusername/HurufinApp.git
   cd HurufinApp
   ```

2. **Buat virtual environment** (direkomendasikan):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # atau
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verifikasi instalasi Tesseract**:
   - Cek apakah Tesseract terinstall di `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - Jika tidak, update path di kode sesuai lokasi instalasi

## Cara Penggunaan

### Quick Start

```bash
python main.py
```

### Workflow Lengkap

#### Method 1: Tesseract OCR (Traditional)
1. **Load Gambar**: Klik "Masukkan Gambar" untuk pilih file
2. **Auto Process**: Klik "OCR Tesseract" di menu untuk proses otomatis
3. **View Results**: Lihat hasil di panel "Ekstraksi Fitur"

#### Method 2: Custom ML Pipeline (Advanced)
1. **Load Gambar**: Klik "Masukkan Gambar"
2. **Step 1**: Klik "Langkah 1" untuk preprocessing ML
3. **Step 2**: Klik "Langkah 2" untuk segmentasi karakter
4. **Step 3**: Load ML model (otomatis)
5. **Recognition**: Klik "Recognition" untuk hasil akhir

### Format File Didukung

- **Gambar**: PNG, JPG, JPEG, BMP, TIFF, GIF
- **Output**: Plain text, dapat dicopy ke clipboard

---

## Konfigurasi

### Tesseract Path Setup

Jika Tesseract tidak terdeteksi otomatis, update path di:

```python
# src/image_processing/ocr.py
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Parameter Tuning

Edit parameter preprocessing di `src/image_processing/preprocessing.py`:

```python
# Contoh parameter yang bisa disesuaikan
GAUSSIAN_KERNEL_SIZE = 3
MEDIAN_KERNEL_SIZE = 3
MORPHOLOGICAL_KERNEL = (2, 2)
```

---

## Development

### Menambahkan Fitur Baru

1. **Image Processing**: Extend `src/image_processing/preprocessing.py`
2. **OCR Engine**: Modify `src/image_processing/ocr.py`
3. **GUI Components**: Update `src/gui/main_window.py`
4. **ML Models**: Add to `src/alphabetic_recognition/`

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_preprocessing.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Linting
flake8 src/
pylint src/

# Formatting
black src/
isort src/
```

---

## Troubleshooting

### Issues Umum

#### TesseractNotFoundError
```bash
# Solusi 1: Install Tesseract
# Windows: Download dari UB-Mannheim
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract

# Solusi 2: Set Path Manual
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
```

#### ModuleNotFoundError
```bash
# Install missing modules
pip install -r requirements.txt --upgrade

# Force reinstall
pip install --force-reinstall opencv-python PyQt5
```

#### Poor OCR Results
- **Preprocessing**: Sesuaikan parameter blur dan threshold
- **Image Quality**: Gunakan gambar dengan resolusi minimal 300 DPI
- **Language**: Set bahasa Tesseract sesuai konten
- **Character Training**: Retrain custom ML model dengan dataset lebih besar

### Performance Tips

1. **Memory Optimization**: Gunakan batch processing untuk gambar besar
2. **Speed**: Aktifkan GPU acceleration untuk OpenCV (jika tersedia)
3. **Accuracy**: Combine hasil Tesseract + Custom ML untuk akurasi terbaik

---

## Kontribusi

### Cara Berkontribusi

1. **Fork** repository ini
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests untuk fitur baru
- Update dokumentasi sesuai perubahan
- Test di multiple Python versions (3.7, 3.8, 3.9+)

### Areas for Contribution

- [ ] **Language Support**: Tambah bahasa selain Indonesia/English
- [ ] **ML Models**: Improve character recognition accuracy
- [ ] **GUI Enhancement**: Modern dark theme, better UX
- [ ] **Performance**: GPU acceleration, parallel processing
- [ ] **Documentation**: Video tutorials, API docs

---

## Lisensi

Proyek ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail.

```text
MIT License - Copyright (c) 2024 HurufinApp
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## Acknowledgments

### Core Technologies
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** - Google's OCR engine
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[PyQt5](https://www.riverbankcomputing.com/software/pyqt/)** - GUI framework
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning toolkit

### Contributors
- **Development Team** - Initial work dan maintenance
- **Community** - Bug reports, feature requests, dan improvements

---

## Roadmap

### Version 2.0 (Coming Soon)
- [ ] **Real-time OCR** - Webcam integration
- [ ] **Batch Processing** - Multiple file support
- [ ] **Cloud Integration** - Google Cloud Vision API
- [ ] **Mobile App** - Android/iOS companion
- [ ] **REST API** - Web service integration

### Long-term Goals
- [ ] **Multi-language Support** - 20+ languages
- [ ] **Deep Learning** - BERT/Transformer models
- [ ] **Document Analysis** - Layout detection, table extraction
- [ ] **Commercial Features** - Enterprise deployment options

---

**⭐ Star this repository if you find it helpful!**

**📧 Questions? Open an [issue](https://github.com/yourusername/HurufinApp/issues) or contact us!**
