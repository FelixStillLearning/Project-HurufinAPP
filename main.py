"""
Titik awal aplikasi Hurufin OCR.
Santai aja, ini file utama buat jalanin aplikasinya :)
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

# Tambahin direktori src ke Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import HurufinMainWindow
from utils.constants import *
from utils.helpers import setup_logging


def main():
    """Titik masuk utama aplikasi."""
    # Setup logging biar gampang tracking error/info
    logger = setup_logging("INFO")
    logger.info("Memulai Aplikasi Hurufin OCR")
    
    # Bikin QApplication (wajib buat aplikasi PyQt)
    app = QApplication(sys.argv)
    
    # Set icon aplikasi
    app.setWindowIcon(QIcon(LOGO_DIR))
    
    # Bikin dan tampilkan main window
    window = HurufinMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    
    window.setWindowIcon(QIcon(LOGO_DIR))
    
    window.show()
    
    logger.info("Aplikasi berhasil dijalankan")
    
    # Jalankan aplikasi
    try:
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Error aplikasi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
