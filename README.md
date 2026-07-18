# God Eye — Gesture Mind

**Body Language OS Controller**: kontrol komputer kamu secara real-time hanya dengan gerakan tangan dan ekspresi wajah, langsung dari webcam. Aplikasi ini mendeteksi gesture tangan (menggunakan MediaPipe) untuk menjalankan perintah sistem operasi (scroll, volume, screenshot, lock screen, dll), sekaligus menganalisis wajah (emosi, perkiraan usia, dan gender) menggunakan DeepFace. Semua ditampilkan dalam tampilan HUD bergaya terminal/cyber di atas feed kamera.

Project ini adalah single-file Python script: `gesture_control.py`.

## Fitur Utama

- **Kontrol OS via gesture tangan** (real-time, via webcam):
  - ✊ Kepalan tangan → Tutup aplikasi aktif (Alt+F4)
  - ✌ Peace / V-sign → Ambil screenshot
  - ☝ Telunjuk ke atas → Volume Up
  - 👇 Telunjuk ke bawah → Volume Down
  - 🖐 Telapak tangan tinggi/rendah → Scroll Up/Down otomatis
  - 👍 Jempol ke atas → Aksi "like" (tidak memicu perintah)
  - 👎 Jempol ke bawah → Tutup aplikasi
  - 👌 OK sign → Play/Pause media
  - 🤟 Dua tangan terdeteksi → Lock screen (Windows/macOS/Linux)
- **Analisis wajah** (setiap beberapa detik, berjalan di thread terpisah agar tidak mengganggu FPS):
  - Deteksi emosi dominan (happy, sad, angry, surprised, neutral, fear, disgust) beserta skor per-emosi
  - Estimasi usia dan gender
- **Live HUD overlay** bergaya terminal/cyber: panel info gesture terdeteksi + confidence, bar skor emosi, estimasi usia/gender, log event, panduan gesture, skeleton tangan, bounding box wajah, dan efek scan-line.
- Cooldown per-gesture agar aksi tidak terpicu berulang kali secara tidak sengaja.
- DeepFace bersifat opsional — jika belum terinstall, fitur analisis wajah otomatis dinonaktifkan dan sisanya (kontrol gesture) tetap berjalan.

## Tech Stack

- **Python 3**
- [OpenCV](https://opencv.org/) (`opencv-python`) — capture kamera & rendering HUD
- [MediaPipe](https://developers.google.com/mediapipe) — deteksi landmark tangan, wajah, dan pose
- [DeepFace](https://github.com/serengil/deepface) — analisis emosi, usia, dan gender (opsional)
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — eksekusi perintah OS (scroll, volume, alt+F4, screenshot, dll)
- NumPy

## Instalasi

1. Clone repo ini:
   ```bash
   git clone https://github.com/Dhiyaahaq33/god-eye.git
   cd god-eye
   ```
2. (Disarankan) buat virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate # macOS/Linux
   ```
3. Install dependensi:
   ```bash
   pip install opencv-python mediapipe deepface numpy pyautogui pillow
   ```
   > Catatan: jika tidak ingin menggunakan fitur analisis wajah (emosi/usia/gender), `deepface` boleh dilewati — aplikasi tetap berjalan dengan fitur kontrol gesture saja.

## Cara Menjalankan

```bash
python gesture_control.py
```

- Pastikan webcam aktif dan tidak dipakai aplikasi lain.
- Jendela HUD akan muncul menampilkan feed kamera + panel informasi di sisi kanan.
- Tekan **Q** atau **ESC** untuk keluar dari aplikasi.

## Konfigurasi

Aplikasi ini berjalan sepenuhnya secara lokal menggunakan webcam perangkat dan **tidak memerlukan API key, token, atau environment variable apa pun**. Tidak ada file `.env` yang dibutuhkan untuk menjalankan `gesture_control.py`.

## Peringatan

Aplikasi ini dapat mengeksekusi perintah nyata di sistem operasi (menutup aplikasi aktif via Alt+F4, mengunci layar, mengubah volume, dll) berdasarkan gesture yang terdeteksi dari kamera. Gunakan dengan hati-hati, terutama saat menguji gesture baru, agar tidak secara tidak sengaja menutup pekerjaan yang belum disimpan.

## Lisensi

MIT License — lihat file [LICENSE](LICENSE).
