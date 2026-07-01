# ==============================================================================
# PNEUSCAN BI - HOSPITAL AI DASHBOARD (STREAMLIT CLINICAL VERSION)
# ==============================================================================

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.graph_objects as go

# 1. ATURAN HALAMAN UTAMA (TEMA GELAP BERSIH)
st.set_page_config(
    page_title="PneuScan BI - Hospital Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk memperjelas teks dan mempercantik tampilan slider
st.markdown("""
    <style>
        .stSlider [data-baseweb="slider"] {
            margin-bottom: 25px;
        }
        .stSlider [data-testid="stMarkdownContainer"] {
            color: #ffffff !important;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# 2. MEMUAT MODEL AI (Menggunakan Cache agar Ringan saat Online)
@st.cache_resource
def load_hospital_model():
    try:
        return tf.keras.models.load_model('pneumonia_model.keras')
    except:
        return None

model = load_hospital_model()

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "L":
        image = ImageOps.grayscale(image)
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. HEADER DASHBOARD RUMAH SAKIT
st.title("🏥 Hospital AI Diagnostics System")
st.caption("Integrasi Deep Learning untuk Skrinning Kesehatan Paru-Paru Pasien & Manajemen Rumah Sakit")
st.markdown("---")

# 4. PEMBAGIAN KOLOM UTAMA
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 📥 Data & Parameter")
    uploaded_file = st.file_uploader("Unggah Foto Rontgen Pasien (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    st.markdown("### ⚙️ Konfigurasi Medis")
    # Slider dibuat pas di posisi 0.40 rekomendasi skrinning medis awal
    threshold = st.slider(
        "Ambang Batas Sensitivitas (Threshold):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.40, 
        step=0.05
    )
    st.markdown("<p style='color: #cbd5e1; font-size: 13px; font-style: italic;'>Catatan Klinis: Angka 0.40 digunakan untuk meningkatkan sensitivitas agar infeksi dini tidak lolos.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💻 Status Sistem")
    if model is not None:
        st.success("Otak AI Model: **SIAP**")
    else:
        st.error("Otak AI Model: **BELUM TERDETEKSI**")
    st.caption("Arsitektur: Custom CNN Classifier")

with col2:
    st.markdown("### 📋 Laporan Analisis & Rekomendasi Klinis")
    
    if uploaded_file is not None:
        if model is not None:
            # Membaca dan memproses gambar rontgen
            image = Image.open(uploaded_file)
            processed_img = preprocess_image(image)
            
            # Prediksi dari model AI
            prediction = model.predict(processed_img)[0][0]
            
            # Perhitungan probabilitas standar (1 = Pneumonia, 0 = Normal)
            prob_pneumonia = float(prediction) * 100
            prob_normal = (1.0 - float(prediction)) * 100
            
            # PERBAIKAN TOTAL URUTAN LOGIKA: Jika nilai prediksi di atas threshold = PNEUMONIA
            if prediction >= threshold:
                st.error("## ⚠️ TERINDIKASI INFEKSI PNEUMONIA")
                st.markdown("##### **Sistem AI mendeteksi adanya akumulasi cairan atau bercak opasitas pada jaringan paru-paru pasien.**")
                st.markdown(
                    """
                    <div style="background-color: #1e293b; padding: 15px; border-left: 5px solid #ef4444; border-radius: 6px;">
                        <span style="color: #cbd5e1; font-style: italic; font-size: 14px;">
                            Rekomendasi Klinis: Segera konfirmasikan hasil skrinning awal AI ini dengan Dokter Spesialis Paru (Pulmonolog) untuk penanganan medis lebih lanjut.
                        </span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.success("## ✅ PARU-PARU NORMAL & SEHAT")
                st.markdown("##### **Sistem AI menunjukkan bahwa kerapatan struktur dan ruang udara paru-paru pasien bersih dalam batas normal.**")
                st.markdown(
                    """
                    <div style="background-color: #1e293b; padding: 15px; border-left: 5px solid #10b981; border-radius: 6px;">
                        <span style="color: #cbd5e1; font-style: italic; font-size: 14px;">
                            Rekomendasi Klinis: Kondisi sistem pernapasan baik. Tetap pertahankan pola hidup sehat dan lakukan pemeriksaan berkala jika diperlukan.
                        </span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            st.markdown("---")
            
            # Pembagian sub-kolom di dalam Kolom Kanan (Gambar vs Grafik Donat)
            sub_col1, sub_col2 = st.columns([1, 1])
            
            with sub_col1:
                st.markdown("<h5 style='text-align: center; color: #ffffff;'>📸 Citra Rontgen Pasien</h5>", unsafe_allow_html=True)
                
                # Memberikan warna border dinamis pada gambar sesuai hasil diagnosis
                border_color = "#ef4444" if prediction >= threshold else "#10b981"
                st.markdown(
                    f"""
                    <div style="text-align: center; border: 2px solid {border_color}; padding: 5px; border-radius: 8px; background-color: #0f172a;">
                        <img src="data:image/png;base64," style="display: none;" />
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.image(image, use_container_width=True)
                st.caption(f"Nama Berkas: {uploaded_file.name}")
                
            with sub_col2:
                st.markdown("<h5 style='text-align: center; color: #ffffff;'>📊 Komposisi Hasil Analisis AI</h5>", unsafe_allow_html=True)
                
                # VISUALISASI UTAMA: GRAFIK DONAT YANG RAMAH PUBLIK
                fig_donut = go.Figure(go.Pie(
                    labels=['Normal (Sehat)', 'Pneumonia (Infeksi)'],
                    values=[prob_normal, prob_pneumonia],
                    hole=.6,
                    marker_colors=['#10b981', '#ef4444'],
                    textinfo='percent',
                    textfont=dict(size=14, weight='bold'),
                    hoverinfo='label+percent'
                ))
                fig_donut.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=250,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.error("Gagal melakukan analisis karena file model 'pneumonia_model.keras' tidak ditemukan di folder.")
    else:
        st.info("Menunggu berkas foto rontgen dada pasien diunggah pada panel sebelah kiri.")