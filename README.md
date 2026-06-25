# E-Commerce Customer Behavior Analytics for Purchase Prediction 

## Project Overview

Project ini bertujuan membangun sistem prediksi potensi pembelian pelanggan pada platform e-commerce berdasarkan aktivitas yang dilakukan selama customer journey. Model prediksi dikembangkan menggunakan algoritma Random Forest dan Artificial Neural Network (ANN) untuk membandingkan performa Machine Learning dan Deep Learning dalam menyelesaikan permasalahan klasifikasi pelanggan.

Model terbaik kemudian diimplementasikan ke dalam aplikasi Streamlit sehingga pengguna dapat melakukan prediksi peluang pembelian pelanggan secara interaktif berdasarkan aktivitas yang dilakukan selama menggunakan platform e-commerce.

## Latar Belakang Project
Perusahaan e-commerce memiliki jutaan data aktivitas pelanggan yang berasal dari berbagai interaksi, seperti pencarian produk, melihat detail produk, membaca ulasan, menambahkan produk ke keranjang, hingga melakukan checkout. Namun, tidak seluruh aktivitas tersebut berakhir menjadi transaksi pembelian.

Tanpa adanya analisis perilaku pelanggan, perusahaan akan kesulitan mengidentifikasi pelanggan yang benar-benar memiliki potensi untuk membeli. Akibatnya, strategi pemasaran seperti pemberian promo, voucher, maupun notifikasi sering kali diberikan secara umum kepada seluruh pelanggan sehingga kurang efektif dan meningkatkan biaya pemasaran.

Oleh karena itu, perusahaan membutuhkan sistem berbasis Machine Learning yang mampu menganalisis pola aktivitas pelanggan dan memprediksi kemungkinan pelanggan melakukan pembelian. Informasi tersebut dapat dimanfaatkan untuk mendukung pengambilan keputusan yang lebih tepat sasaran, seperti personalisasi promosi, reminder checkout, maupun strategi retensi pelanggan.

## Tujuan Project
- Menganalisis perilaku pelanggan berdasarkan aktivitas pada platform e-commerce.
- Mengidentifikasi aktivitas pelanggan yang paling berpengaruh terhadap keputusan pembelian.
- Membangun model prediksi potensi pembelian menggunakan algoritma Random Forest dan Artificial Neural Network (ANN).
- Membandingkan performa Machine Learning dan Deep Learning menggunakan berbagai metrik evaluasi.
- Mengimplementasikan model terbaik ke dalam aplikasi Streamlit sehingga dapat digunakan untuk melakukan prediksi secara real-time.
- Memberikan rekomendasi bisnis berbasis hasil prediksi guna membantu perusahaan meningkatkan efektivitas strategi pemasaran dan peluang konversi pelanggan.

## Cara Menjalankan Project
1. Membuat Environment Python

Buka Anaconda Prompt, lalu buat environment baru dengan Python 3.10 agar library yang digunakan lebih stabil.

```bash
conda create -n take_home python=3.10 -y
```

2. Mengaktifkan Environment

Setelah environment berhasil dibuat, aktifkan environment tersebut dengan perintah berikut:

```bash
conda activate take_home
```

3. Masuk ke Folder Project

Arahkan terminal ke folder tempat final project disimpan.

```bash
cd /d d:\Dokumen Bootcamp Dibimbing 2025\[WAJIB] Final Take Home Test\[WAJIB] File Project\Take Home Test_Raihan Azhar Rafi
```

Jika folder project berada di lokasi lain, sesuaikan path folder sesuai lokasi penyimpanan di laptop masing-masing.

4. Membuka Project di Visual Studio Code

Setelah berada di folder project, buka Visual Studio Code dengan perintah:

```bash
code .
```

5. Memastikan Environment Aktif di VS Code

Di Visual Studio Code, buka terminal baru melalui menu:

Terminal > New Terminal

Pastikan terminal masih menggunakan environment:

```bash
(take_home)
```

Jika belum aktif, jalankan kembali:

```bash
conda activate take_home
```

6. Install Library yang Dibutuhkan

Install seluruh dependency project menggunakan file Requirements.txt.

```bash
pip install -r Requirements.txt
```

Tunggu sampai seluruh proses instalasi selesai.

7. Menjalankan Aplikasi Streamlit

Setelah semua library terinstall, jalankan aplikasi dengan perintah:

```bash
streamlit run Streamlit/App.py
```

Setelah perintah dijalankan, aplikasi akan terbuka otomatis di browser. Jika tidak terbuka otomatis, salin link lokal yang muncul di terminal, biasanya seperti:

```text
http://localhost:8501
```
