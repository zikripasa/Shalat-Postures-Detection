# Panduan pengguna Salat Vision

*(English summary: open the web app, upload or record a side-on video of one person praying, press **Analisis video**,
then read the phase timeline. Grey segments are uncertain. The system recognises postures; it does not judge prayers.)*

## 1. Menyiapkan rekaman

* Letakkan kamera **menyamping**, setinggi pinggang sampai dada, sekitar 2–4 meter dari orang yang shalat.
* Pastikan **seluruh tubuh** terlihat dari kepala sampai kaki, termasuk saat sujud.
* Satu orang saja di dalam bingkai. Cahaya cukup, latar tidak terlalu ramai.
* Format MP4, MOV, atau WebM, paling besar 150 MB dan 10 menit.

## 2. Menganalisis video

1. Buka **Analisis**. Pilih **Unggah berkas**, **Rekam dengan kamera**, atau **Pakai video demo**.
2. Pilih kerapatan analisis: 2, 5 (disarankan), atau 10 frame per detik.
3. Biarkan **Buat video beranotasi** tercentang jika ingin video dengan kerangka tubuh dan label fase.
4. Tekan **Analisis video**. Bilah kemajuan menunjukkan unggahan, lalu jumlah frame yang sudah dianalisis.

Rekaman kamera membutuhkan izin kamera dan halaman yang dibuka lewat `localhost` atau HTTPS.

## 3. Membaca hasil

* **Linimasa fase**: setiap blok adalah satu segmen. Klik blok untuk melompat ke bagian video tersebut.
* **Abu-abu / Tidak pasti**: keyakinan model di bawah 55% atau orang tidak terlihat. Periksa bagian itu secara manual.
* **Dasar penentuan**: *Terlihat langsung* (rukuk, sujud) atau *Terlihat + urutan gerakan* (berdiri, iktidal, duduk,
  tasyahud). Iktidal dan tasyahud tidak dilihat langsung; keduanya disimpulkan dari urutan.
* **Siklus rakaat teramati**: jumlah sujud kedua yang terdeteksi. Ini hitungan deskriptif, bukan penilaian.
* **Bandingkan dengan prediksi mentah**: menunjukkan label per frame sebelum pemrosesan urutan.
* **Unduh**: laporan JSON lengkap, daftar segmen CSV, dan video beranotasi.

## 4. Menganalisis foto

Buka tab **Foto**, pilih foto satu orang. Hasilnya: postur yang dikenali, skor per postur, kerangka tubuh, dan dari mana
lokasi orang ditemukan.

## 5. Privasi

Video diproses di server Salat Vision Anda sendiri, tanpa layanan AI eksternal. Berkas unggahan dihapus setelah
analisis, dan hasil kedaluwarsa setelah 30 menit.

## 6. Batasan

Model dilatih dengan data yang sangat sedikit (40 orang). Hasil bisa keliru pada sudut kamera, pakaian, atau ruangan
yang berbeda. Salat Vision **tidak** menilai sah atau tidaknya shalat.
