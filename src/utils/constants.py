"""
Konstanta yang digunakan di seluruh aplikasi Hurufin.
"""

# Informasi aplikasi
APP_NAME = "Hurufin"
VERSION = "1.0.7"
AUTHOR = "Kelompok 3 PCD - Hurufin"

# Format gambar yang didukung
SUPPORTED_IMAGE_FORMATS = [
    "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.gif"
]

# Parameter default pemrosesan
DEFAULT_MEDIAN_KERNEL_SIZE = 3
DEFAULT_MORPHOLOGY_KERNEL_SIZE = (3, 3)

# Konfigurasi OCR
DEFAULT_TESSERACT_CONFIG = ''
TESSERACT_DEFAULT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Konstanta UI
WINDOW_TITLE = f"{APP_NAME} v{VERSION}"
DEFAULT_WINDOW_SIZE = (1200, 800)

# Path file
DATA_DIR = "data"
SAMPLE_IMAGES_DIR = "data/sample_images"
TEST_IMAGES_DIR = "data/test_images"
CONFIG_DIR = "config"
DOCS_DIR = "docs"
TESTS_DIR = "tests"
LOGO_DIR = "data/Hurufin/logo.png"
