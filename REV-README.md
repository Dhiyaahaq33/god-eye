# MATA DEWA
### GESTURE MIND — Body Language OS Controller (D:\BOT\GOD EYE)

---

## Apa Ini?

Ini adalah program yang bikin kamu bisa mengontrol komputer cuma pakai gerakan tangan dan ekspresi wajah lewat webcam — tanpa mouse, tanpa keyboard. Angkat telapak tangan buat scroll, kepalkan tangan buat tutup aplikasi, buat tanda victory (V) buat screenshot, dan banyak lagi. Program ini juga bisa membaca ekspresi wajah (senang, sedih, marah, dll) serta menebak usia dan gender orang di depan kamera.

Cocok buat demo teknologi computer vision, atau buat yang mau kontrol PC secara "futuristik" tanpa sentuhan.

## Fitur Utama

- Deteksi gesture tangan pakai model resmi Google MediaPipe (GestureRecognizer), dengan fallback otomatis ke perhitungan manual kalau model gagal dimuat
- Scroll otomatis (telapak tangan terbuka, gerak naik/turun)
- Tutup aplikasi otomatis (kepalan tangan / jempol ke bawah)
- Volume naik/turun (jari telunjuk arah atas/bawah)
- Screenshot otomatis (tanda "V" / Victory)
- Minimize semua jendela (gesture "I Love You")
- Lock screen (kalau dua tangan terlihat kamera bersamaan)
- Pause/Play media (gesture "OK")
- Analisis ekspresi wajah — deteksi senang, sedih, marah, terkejut, netral, takut, jijik
- Deteksi perkiraan usia & gender
- Tampilan HUD (Head-Up Display) real-time bergaya terminal, bisa fullscreen (tombol F)

## Teknologi yang Dipakai

- Python 3.11
- OpenCV (opencv-python) — proses gambar dari webcam
- MediaPipe — deteksi gesture tangan
- DeepFace + TensorFlow — analisis wajah, usia, gender
- PyAutoGUI — mengontrol aksi komputer (scroll, volume, dll)
- Pillow

## Cara Instalasi

1. Python 3.11 sudah tersedia di laptop ini — tidak perlu install ulang.
2. Buka folder proyek:
   ```powershell
   cd "D:\BOT\GOD EYE"
   ```
3. Buat virtual environment khusus (SANGAT disarankan untuk proyek ini karena versi library-nya sudah dikunci ketat):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependency PERSIS sesuai versi di `requirements.txt` (jangan upgrade manual, karena kombinasi versi ini sudah teruji jalan bareng di Windows + Python 3.11):
   ```powershell
   pip install -r requirements.txt
   ```
   Catatan dari developer proyek ini: JANGAN install `mediapipe`/`tensorflow`/`protobuf` secara terpisah tanpa versi yang sudah dipin, karena bisa menarik versi TensorFlow terbaru yang butuh protobuf baru dan malah merusak fitur MediaPipe.
5. Webcam harus aktif dan diizinkan diakses oleh aplikasi Python (cek di Windows Settings > Privacy > Camera).
6. Saat pertama kali dijalankan, model GestureRecognizer (~8MB) akan otomatis di-download ke folder `models/` — pastikan koneksi internet aktif saat run pertama.

## Cara Menjalankan

```powershell
.venv\Scripts\activate
python gesture_control.py
```

Tombol `F` untuk toggle fullscreen HUD, tutup program dengan gesture kepalan tangan atau tombol keyboard sesuai petunjuk di layar.


## Catatan Penting

- Tidak ditemukan file `.env` atau API key di folder ini — semua proses jalan lokal di komputer, tidak ada koneksi ke layanan cloud berbayar, jadi aman dari sisi kebocoran kredensial.
- Folder `screenshots/` menyimpan hasil screenshot otomatis dari gesture "Victory" — cek isinya kalau folder makin besar ukurannya, mungkin perlu dibersihkan berkala.
- Versi library di `requirements.txt` sengaja dikunci ketat (opencv-python <5.0, mediapipe 0.10.9, protobuf 3.20.3, dst) — kalau nanti ada error setelah update Windows/Python, kemungkinan besar karena versi library berubah, bukan karena kode-nya rusak.

## Kebutuhan API LLM

- **Butuh API LLM?** Tidak — semua deteksi (gesture tangan, ekspresi wajah, usia/gender) pakai model computer vision (MediaPipe, DeepFace) yang kerjanya klasifikasi angka/koordinat, bukan memahami atau menghasilkan bahasa alami. Tidak ada teks bebas yang perlu diringkas, ditulis, atau diklasifikasi di sini.
- **Bisa pakai API Claude (Anthropic)?** Tidak relevan — proyek ini tidak memproses bahasa alami / tidak butuh LLM sama sekali. Kalau suatu saat mau ditambah fitur voice command ("matikan aplikasi", dst) yang perlu dimengerti secara bebas, baru Claude Haiku 4.5 bisa dipakai sebagai lapisan interpretasi perintah — tapi itu di luar fitur yang ada sekarang.

## Instalasi & Eksekusi Offline

- **Bisa instalasi offline?** Sebagian — install pertama kali WAJIB online karena `pip install -r requirements.txt` narik banyak library besar (MediaPipe, TensorFlow, OpenCV, dll) dari internet, dan run pertama juga otomatis download model GestureRecognizer (~8MB) ke folder `models/`. Kalau semua package dan model itu sudah pernah didownload sekali di komputer yang sama, instalasi ulang bisa pakai cache pip lokal (offline).
- **Bisa dijalankan offline (setelah terinstall)?** Ya, sepenuhnya — setelah dependency dan model gesture sudah ada di lokal, `python gesture_control.py` cuma memproses feed webcam lewat OpenCV/MediaPipe/DeepFace di komputer sendiri, tidak ada panggilan API cloud atau LLM apapun. Bisa dipakai penuh tanpa internet.
