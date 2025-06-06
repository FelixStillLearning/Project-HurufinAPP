# Aplikasi Hurufin OCR - Panduan Pengguna

## Daftar Isi
1. [Mulai](#mulai)
2. [Gambaran Antarmuka](#gambaran-antarmuka)
3. [Langkah-Langkah Penggunaan](#langkah-langkah-penggunaan)
4. [Tips untuk Hasil Lebih Baik](#tips-untuk-hasil-lebih-baik)
5. [Pemecahan Masalah](#pemecahan-masalah)

## Mulai

### Kebutuhan Sistem
- Windows 10 atau versi lebih baru
- Python 3.7 atau versi lebih tinggi
- Tesseract OCR sudah terinstal
- Rekomendasi RAM minimal 4GB

### Pengaturan Awal
1. Pastikan semua dependensi sudah terinstal (lihat README.md)
2. Jalankan aplikasi dengan perintah `python main.py`
3. Jendela utama akan muncul dengan tiga area tampilan gambar

## Gambaran Antarmuka

Antarmuka Hurufin terdiri dari:

### Area Utama
- **Tampilan Gambar Asli**: Menampilkan gambar yang dimuat
- **Tampilan Pra-pemrosesan**: Menampilkan gambar setelah langkah pra-pemrosesan
- **Tampilan Ekstraksi**: Menampilkan gambar yang diproses akhir dengan kotak teks terdeteksi
- **Area Keluaran Teks**: Menampilkan teks yang diekstrak

### Tombol Kontrol
- **Input**: Memuat file gambar untuk diproses
- **Langkah 1**: Terapkan pra-pemrosesan (grayscale, blur, perenggangan kontras)
- **Langkah 2**: Terapkan operasi morfologi (thresholding, pembukaan)
- **Pengenalan**: Lakukan OCR dan ekstrak teks

## Langkah-Langkah Penggunaan

### 1. Memuat Gambar
1. Klik tombol **Input**
2. Telusuri dan pilih file gambar
3. Format yang didukung: PNG, JPG, JPEG, BMP, TIFF, GIF
4. Gambar akan muncul di area tampilan "Gambar Asli"

### 2. Pra-pemrosesan (Langkah 1)
1. Klik **Langkah 1** untuk menerapkan pra-pemrosesan
2. Ini melakukan:
   - Konversi ke grayscale
   - Penyaringan blur median (menghilangkan noise)
   - Perenggangan kontras (meningkatkan kontras)
3. Gambar yang diproses muncul di area tampilan "Pra-pemrosesan"

### 3. Operasi Morfologi (Langkah 2)
1. Klik **Langkah 2** untuk menerapkan pemrosesan lanjutan
2. Ini melakukan:
   - Thresholding Otsu (mengubah ke gambar biner)
   - Pembukaan morfologi (menghilangkan noise kecil)
3. Gambar yang diproses akhir muncul di area tampilan "Ekstraksi"

### 4. Pengenalan Teks
1. Klik **Pengenalan** untuk mengekstrak teks
2. Sistem akan:
   - Menganalisis gambar yang diproses
   - Mendeteksi kotak pembatas karakter
   - Mengekstrak teks menggunakan OCR
3. Hasil muncul di area keluaran teks
4. Kotak pembatas digambar di sekitar karakter yang terdeteksi

## Tips untuk Hasil Lebih Baik

### Kualitas Gambar
- Gunakan gambar resolusi tinggi (300 DPI atau lebih)
- Pastikan kontras yang baik antara teks dan latar belakang
- Hindari gambar yang buram atau terdistorsi
- Luruskan teks yang miring jika memungkinkan

### Karakteristik Teks
- Paling baik digunakan dengan teks cetak
- Font standar yang jelas lebih baik daripada font dekoratif
- Hindari gambar dengan latar belakang yang kompleks
- Pastikan teks tidak terlalu kecil (minimum 12pt disarankan)

### Penyesuaian Pra-pemrosesan
Jika hasilnya buruk, pertimbangkan:
- Menggunakan gambar sumber yang berbeda
- Menyesuaikan kecerahan/kontras gambar sebelum dimuat
- Mencoba gambar dengan kondisi pencahayaan yang lebih baik

## Pemecahan Masalah

### Tidak Ada Teks yang Terdeteksi
**Kemungkinan Penyebab:**
- Kontras gambar terlalu rendah
- Teks terlalu kecil atau buram
- Latar belakang yang kompleks mengganggu teks

**Solusi:**
- Coba gambar dengan kontras yang lebih baik
- Gunakan gambar dengan resolusi lebih tinggi
- Pastikan teks terlihat jelas

### Pengenalan Teks yang Salah
**Kemungkinan Penyebab:**
- Kualitas gambar yang buruk
- Font atau pemformatan yang tidak biasa
- Ketidakcocokan bahasa

**Solusi:**
- Gunakan gambar yang lebih jelas dan berkualitas lebih tinggi
- Coba font standar jika memungkinkan
- Periksa apakah bahasa teks sesuai dengan pengaturan OCR

### Aplikasi Mengalami Crash
**Kemungkinan Penyebab:**
- Dependensi yang hilang
- File gambar yang korup
- Sumber daya sistem yang tidak mencukupi

**Solusi:**
- Verifikasi semua persyaratan telah terinstal
- Coba file gambar yang berbeda
- Tutup aplikasi lain untuk membebaskan memori

### Masalah Kinerja
**Kemungkinan Penyebab:**
- File gambar besar
- Sumber daya sistem yang tidak mencukupi
- Beberapa aplikasi berjalan bersamaan

**Solusi:**
- Ubah ukuran gambar besar sebelum diproses
- Tutup aplikasi yang tidak perlu
- Proses gambar satu per satu

## Penggunaan Lanjutan

### Pemrosesan Batch
Untuk beberapa gambar:
1. Proses gambar satu per satu
2. Salin teks yang diekstrak ke dokumen eksternal
3. Ulangi untuk setiap gambar

### Menyimpan Hasil
- Salin teks dari area keluaran
- Tempelkan ke editor teks atau dokumen
- Simpan sesuai kebutuhan

### Bekerja dengan Berbagai Bahasa
- Konfigurasi default berfungsi dengan bahasa Inggris
- Untuk bahasa lain, paket bahasa Tesseract mungkin diperlukan
- Hubungi administrator sistem untuk instalasi paket bahasa

## Praktik Terbaik

1. **Mulai dengan gambar berkualitas tinggi** - ini adalah faktor terpenting
2. **Ikuti langkah pemrosesan sesuai urutan** - setiap langkah dibangun di atas langkah sebelumnya
3. **Tinjau hasil pra-pemrosesan** - pastikan gambar terlihat baik sebelum OCR
4. **Periksa keluaran teks dengan cermat** - OCR mungkin membuat kesalahan dengan karakter yang mirip
5. **Simpan pekerjaan Anda secara teratur** - salin hasil teks penting segera

## Mendapatkan Bantuan

Jika Anda mengalami masalah yang tidak tercakup dalam panduan ini:
1. Periksa log aplikasi di direktori `logs/`
2. Coba dengan gambar yang berbeda untuk mengisolasi masalah
3. Verifikasi semua persyaratan sistem terpenuhi
4. Hubungi tim pengembang dengan pesan kesalahan yang spesifik
