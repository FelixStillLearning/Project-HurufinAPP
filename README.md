# Hurufin OCR Application

Aplikasi OCR (Optical Character Recognition) profesional yang dibangun dengan Python dan PyQt5 untuk memproses teks dari gambar.

## Fitur Unggulan

- **Memuat Gambar**: Mendukung berbagai format gambar (PNG, JPG, JPEG, BMP, TIFF)
- **Pra-pemrosesan Gambar**: 
  - Konversi ke grayscale
  - Filter median blur
  - Peregangan kontras
  - Thresholding Otsu
  - Operasi morfologi
- **Proses OCR**: Ekstraksi teks menggunakan Tesseract OCR
- **Deteksi Bounding Box**: Feedback visual untuk karakter yang terdeteksi
- **GUI Profesional**: Interface PyQt5 yang bersih dan intuitif

## Struktur Proyek

```
HurufinApp/
├── main.py                 # Entry point aplikasi utama
├── requirements.txt        # Dependencies Python
├── README.md              # File ini
├── config/
│   └── settings.py        # Konfigurasi aplikasi
├── data/
│   ├── sample_images/     # Gambar sample untuk testing
│   └── test_images/       # Gambar untuk testing
├── docs/
│   └── user_guide.md      # Dokumentasi pengguna
├── src/
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── GUI.ui         # File UI PyQt5
│   │   └── main_window.py # Implementasi main window
│   ├── image_processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Fungsi pra-pemrosesan gambar
│   │   └── ocr.py          # Fungsi pemrosesan OCR
│   └── utils/
│       ├── __init__.py
│       ├── constants.py    # Konstanta aplikasi
│       └── helpers.py      # Fungsi pembantu
└── tests/
    └── test_preprocessing.py # Unit tests
```

## Instalasi

### Prasyarat

1. **Python 3.7+**: Pastikan Python sudah terinstall di sistem Anda
2. **Tesseract OCR**: Download dan install dari [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Path instalasi default: `C:\Program Files\Tesseract-OCR\`
   - Tambahkan ke system PATH atau update path di `config/settings.py`

### Setup

1. **Clone atau download** proyek ini ke mesin lokal Anda

2. **Buat virtual environment** (direkomendasikan):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Untuk Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verifikasi instalasi Tesseract**:
   - Cek apakah Tesseract terinstall di `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - Jika terinstall di tempat lain, update path di `config/settings.py`

## Cara Penggunaan

### Menjalankan Aplikasi

```bash
python main.py
```

### Menggunakan Aplikasi

1. **Load Gambar**: Klik tombol "Input" untuk memilih file gambar
2. **Preprocessing**: Klik "Step 1" untuk menerapkan pra-pemrosesan gambar
3. **Operasi Morfologi**: Klik "Step 2" untuk menerapkan thresholding dan operasi morfologi
4. **OCR Recognition**: Klik "Recognition" untuk mengekstrak teks dari gambar yang sudah diproses

### Format Gambar yang Didukung

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)

## Konfigurasi

Edit `config/settings.py` untuk kustomisasi:

- **Path Tesseract**: Update `OCR_CONFIG['tesseract_path']`
- **Parameter processing**: Modifikasi `PREPROCESSING_CONFIG`
- **Pengaturan GUI**: Sesuaikan `GUI_CONFIG`
- **Pengaturan bahasa**: Ubah `OCR_CONFIG['language']` untuk bahasa yang berbeda

## Development

### Arsitektur Proyek

Aplikasi ini mengikuti arsitektur modular:

- **GUI Layer** (`src/gui/`): Komponen interface PyQt5
- **Processing Layer** (`src/image_processing/`): Logic inti untuk pemrosesan gambar dan OCR
- **Utilities** (`src/utils/`): Fungsi pembantu dan konstanta
- **Configuration** (`config/`): Pengaturan dan parameter aplikasi

### Menambahkan Fitur Baru

1. **Image Processing**: Tambahkan method baru ke `src/image_processing/preprocessing.py`
2. **Fitur OCR**: Extend `src/image_processing/ocr.py`
3. **Komponen GUI**: Modifikasi `src/gui/main_window.py` dan `GUI.ui`
4. **Konfigurasi**: Update `config/settings.py` untuk parameter baru

### Testing

Jalankan tests menggunakan:
```bash
python -m pytest tests/
```

## Troubleshooting

### Masalah Umum

1. **"ModuleNotFoundError: No module named 'pytesseract'"**
   - Install pytesseract: `pip install pytesseract`

2. **"TesseractNotFoundError"**
   - Install Tesseract OCR
   - Update path di `config/settings.py`

3. **"Failed to load image"**
   - Cek apakah format gambar didukung
   - Pastikan file gambar tidak corrupt

4. **Hasil OCR Kurang Bagus**
   - Coba parameter preprocessing yang berbeda
   - Pastikan gambar memiliki kontras dan resolusi yang baik
   - Pertimbangkan menggunakan model bahasa Tesseract yang berbeda

## Kontribusi

1. Fork repository ini
2. Buat feature branch
3. Buat perubahan Anda
4. Tambahkan tests untuk fungsionalitas baru
5. Submit pull request

## Lisensi

Proyek ini adalah open source dan tersedia di bawah [MIT License](LICENSE).

## Ucapan Terima Kasih

- **Tesseract OCR**: Engine OCR dari Google
- **OpenCV**: Library computer vision
- **PyQt5**: Framework GUI
- **NumPy**: Library komputasi numerik
