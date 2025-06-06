"""
Fungsi pembantu dan utilitas untuk aplikasi Hurufin.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np


def validate_image_path(image_path: str) -> bool:
    """
    Validasi apakah path yang diberikan adalah file gambar yang valid.
    
    Argumen:
        image_path (str): Path ke file gambar
        
    Return:
        bool: True jika file gambar valid, False jika tidak
    """
    if not os.path.exists(image_path):
        return False
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    file_extension = Path(image_path).suffix.lower()
    
    return file_extension in valid_extensions


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Atur konfigurasi logging.
    
    Argumen:
        log_level (str): Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path ke file log. Jika None, log hanya ke konsol.
        
    Return:
        logging.Logger: Instance logger yang sudah dikonfigurasi
    """
    logger = logging.getLogger("Hurufin")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Hapus handler yang ada
    logger.handlers.clear()
    
    # Buat formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler konsol
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler file (jika ditentukan)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_image_info(image_path: str) -> dict:
    """
    Dapatkan informasi dasar tentang file gambar.
    
    Argumen:
        image_path (str): Path ke file gambar
        
    Return:
        dict: Dictionary berisi informasi gambar
    """
    if not validate_image_path(image_path):
        return {}
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        file_size = os.path.getsize(image_path)
        
        return {
            'filename': os.path.basename(image_path),
            'path': image_path,
            'width': width,
            'height': height,
            'channels': channels,
            'file_size': file_size,
            'format': Path(image_path).suffix.upper()[1:]
        }
    except Exception as e:
        print(f"Error getting image info: {e}")
        return {}


def create_directory_structure(base_path: str) -> bool:
    """
    Buat struktur direktori standar untuk aplikasi.
    
    Argumen:
        base_path (str): Path direktori dasar
        
    Return:
        bool: True jika berhasil, False jika gagal
    """
    directories = [
        'data/sample_images',
        'data/test_images',
        'data/output',
        'config',
        'docs',
        'tests',
        'logs'
    ]
    
    try:
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            os.makedirs(full_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory structure: {e}")
        return False


def list_sample_images(sample_dir: str = "data/sample_images") -> List[str]:
    """
    Dapatkan daftar gambar contoh di direktori sample.
    
    Argumen:
        sample_dir (str): Path ke direktori gambar sample
        
    Return:
        List[str]: Daftar path file gambar
    """
    if not os.path.exists(sample_dir):
        return []
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    image_files = []
    
    for filename in os.listdir(sample_dir):
        if Path(filename).suffix.lower() in valid_extensions:
            image_files.append(os.path.join(sample_dir, filename))
    
    return sorted(image_files)


def resize_image_for_display(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Ubah ukuran gambar untuk ditampilkan dengan tetap menjaga rasio aspek.
    
    Argumen:
        image (np.ndarray): Gambar input
        max_width (int): Lebar maksimum
        max_height (int): Tinggi maksimum
        
    Return:
        np.ndarray: Gambar yang sudah diubah ukurannya
    """
    height, width = image.shape[:2]
    
    # Hitung faktor skala
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height, 1.0)  # Jangan upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image
