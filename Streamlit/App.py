# Proses Import Library
import streamlit as st

import pandas as pd
import numpy as np

import joblib

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Proses Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# Proses Load Dataset dan Model
data_model = pd.read_csv(
    BASE_DIR / "Data" / "data_model_final.csv"
)

model_rf = joblib.load(
    BASE_DIR / "Model" / "random_forest.pkl"
)

scaler = joblib.load(
    BASE_DIR / "Model" / "scaler.pkl"
)

# Proses Membuat Sidebar Navigation
menu = st.sidebar.selectbox(
    "📋 Pilih Menu",
    [
        "Dashboard",
        "Customer Prediction",
        "Model Performance",
        "Feature Importance",
        "Business Recommendation",
        "Dataset Overview",
        "About Project"
    ]
)

# Proses Dashboard
if menu == "Dashboard":

    st.title(
        "🛒 Customer Purchase Prediction Dashboard"
    )

    st.markdown(
        """
        Prediksi Potensi Pembelian Pelanggan
        Berdasarkan Aktivitas Pengguna E-Commerce
        """
    )

    total_pelanggan = len(data_model)

    total_beli = (
        data_model["target_pembelian"]
        .sum()
    )

    total_tidak_beli = (
        total_pelanggan
        - total_beli
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Pelanggan",
        total_pelanggan
    )

    col2.metric(
        "Membeli",
        total_beli
    )

    col3.metric(
        "Tidak Membeli",
        total_tidak_beli
    )

# Proses Menambahkan Visualisasi Distribusi Target
    st.subheader(
        "Distribusi Target Pembelian"
    )

    fig = px.pie(
        data_model,
        names="target_pembelian",
        title="Distribusi Pembelian"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Proses Customer Prediction
if menu == "Customer Prediction":

    st.title(
        "🔍 Customer Purchase Prediction"
    )

    jumlah_search = st.number_input(
        "Jumlah Search",
        min_value=0,
        value=1
    )

    jumlah_lihat_produk = st.number_input(
        "Jumlah Lihat Produk",
        min_value=0,
        value=1
    )

    jumlah_baca_review = st.number_input(
        "Jumlah Baca Review",
        min_value=0,
        value=1
    )

    jumlah_keranjang = st.number_input(
        "Jumlah Keranjang",
        min_value=0,
        value=0
    )

    jumlah_checkout = st.number_input(
        "Jumlah Checkout",
        min_value=0,
        value=0
    )

# Proses Membuat Tombol Prediksi
    if st.button(
        "Prediksi Pelanggan"
    ):

        data_input = pd.DataFrame({
            "jumlah_search":[jumlah_search],
            "jumlah_lihat_produk":[jumlah_lihat_produk],
            "jumlah_baca_review":[jumlah_baca_review],
            "jumlah_keranjang":[jumlah_keranjang],
            "jumlah_checkout":[jumlah_checkout]
        })

        data_scaled = scaler.transform(
            data_input
        )

        prediksi = model_rf.predict(
            data_scaled
        )

        probabilitas = (
            model_rf
            .predict_proba(data_scaled)[0][1]
        )

# Proses Menampilkan Hasil Prediksi
        st.subheader(
            "Hasil Prediksi"
        )

        st.metric(
            "Probabilitas Membeli",
            f"{probabilitas:.2%}"
        )

        if probabilitas >= 0.80:

            st.success(
                "High Potential Buyer"
            )

        elif probabilitas >= 0.50:

            st.warning(
                "Medium Potential Buyer"
            )

        else:

            st.error(
                "Low Potential Buyer"
            )

# Proses Model Performance
if menu == "Model Performance":

    st.title(
        "📈 Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        "100%"
    )

    col2.metric(
        "Cross Validation",
        "98.44%"
    )

    col3.metric(
        "AUC Score",
        "1.000"
    )

    st.info(
        """
        Model Random Forest menunjukkan performa yang sangat baik
        dalam membedakan pelanggan yang membeli dan tidak membeli.
        """
    )

# Proses Feature Importance
if menu == "Feature Importance":

    st.title(
        "⭐ Feature Importance"
    )

    importance_df = pd.DataFrame({

        "Fitur": [

            "Jumlah Checkout",
            "Jumlah Keranjang",
            "Jumlah Search",
            "Jumlah Lihat Produk",
            "Jumlah Baca Review"

        ],

        "Importance": [

            0.811,
            0.096,
            0.058,
            0.019,
            0.016

        ]

    })

    fig = px.bar(

        importance_df,

        x="Importance",

        y="Fitur",

        orientation="h",

        title="Feature Importance Random Forest"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Insight Feature Importance
    st.info(
        """
        Fitur jumlah_checkout memiliki nilai importance sebesar 0,811 (81,1%) sehingga menjadi faktor 
        paling dominan dalam model. Hal tersebut menunjukkan bahwa aktivitas checkout merupakan indikator 
        utama yang membedakan pelanggan yang membeli dan tidak membeli.

        Terdapat perbedaan yang cukup signifikan antara jumlah_checkout 0,811 (81,1%) dan jumlah_keranjang 
        0,096 (9,6%). Temuan tersebut menunjukkan bahwa pelanggan yang mencapai tahap checkout memiliki 
        kemungkinan pembelian yang jauh lebih tinggi dibandingkan pelanggan yang hanya menambahkan produk ke 
        keranjang.

        Aktivitas eksplorasi seperti pencarian produk, melihat produk, dan membaca review hanya memberikan 
        kontribusi sekitar 9,3% terhadap keputusan model. Hal tersebut menunjukkan bahwa 3 aktivitas tersebut 
        belum cukup kuat untuk memprediksi pembelian pelanggan secara akurat.

        Berdasarkan hasil analisis tersebut menunjukkan bahwa strategi bisnisnya harus berfokus pada 
        peningkatan jumlah pelanggan yang mencapai tahap checkout, karena berpotensi memberikan dampak 
        terbesar terhadap peningkatan tingkat pembelian pelanggan.
        """
    )

# Proses Business Recommendation
if menu == "Business Recommendation":

    st.title(
        "💡 Business Recommendation"
    )

    st.subheader(
        "Strategi Berdasarkan Potensi Pelanggan"
    )

# High Potential Buyer
    st.success(
        """
        🟢 High Potential Buyer

        • Kirim Voucher Diskon

        • Gratis Ongkir

        • Reminder Checkout

        • Promo Flash Sale
        """
    )

# Medium Potential Buyer
    st.warning(
        """
        🟡 Medium Potential Buyer

        • Rekomendasi Produk

        • Promo Bundling

        • Reminder Produk Favorit
        """
    )

# Low Potential Buyer
    st.error(
        """
        🔴 Low Potential Buyer

        • Awareness Campaign

        • Promosi Umum

        • Iklan Produk Populer
        """
    )

# Menampilkan Pola Aktivitas Pelanggan
    st.subheader(
        "Pola Aktivitas Pelanggan"
    )

    st.markdown(
        """
        Search

        ⬇️

        View Product

        ⬇️

        Review

        ⬇️

        Cart

        ⬇️

        Checkout

        ⬇️

        Purchase
        """
    )

# Proses Dataset Overview
if menu == "Dataset Overview":

    st.title(
        "📊 Dataset Overview"
    )

    st.subheader(
        "Preview Dataset"
    )

    st.dataframe(
        data_model.head()
    )

# Proses Statistik Dataset
    st.subheader(
        "Statistik Dataset"
    )

    st.dataframe(
        data_model.describe()
    )

# Proses Informasi Dataset
    st.subheader(
        "Informasi Dataset"
    )

    info_df = pd.DataFrame({

        "Informasi": [

            "Jumlah Pelanggan",
            "Jumlah Fitur",
            "Target"

        ],

        "Nilai": [

            len(data_model),
            5,
            "target_pembelian"

        ]

    })

    st.dataframe(
        info_df
    )

# Proses Visualisasi Distribusi Target
    st.subheader(
        "Distribusi Target Pembelian"
    )

    fig = px.histogram(

        data_model,

        x="target_pembelian",

        color="target_pembelian"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Proses About Project
if menu == "About Project":

    st.title(
        "ℹ️ About Project"
    )

    st.subheader(
        "Project Background"
    )

    st.write(
        """
        Project ini bertujuan untuk memprediksi potensi pembelian pelanggan
        berdasarkan aktivitas pengguna pada platform e-commerce.
        """
    )

# Proses Business Problem
    st.subheader(
        "Business Problem"
    )

    st.write(
        """
        Perusahaan sering memberikan promosi kepada seluruh pelanggan
        tanpa mengetahui pelanggan mana yang benar-benar memiliki potensi membeli.
        """
    )

# Proses Machine Learning Solution
    st.subheader(
        "Machine Learning Solution"
    )

    st.write(
        """
        Model Random Forest digunakan untuk memprediksi potensi pembelian
        berdasarkan aktivitas Search, View Product, Review, Cart, dan Checkout.
        """
    )

# Proses Penjelasan Tools Yang Digunakan 
    st.subheader(
        "Tools & Technology"
    )

    st.write(
        """
        • Python

        • Pandas

        • Scikit-Learn

        • Random Forest

        • Streamlit

        • Plotly
        """
    )






