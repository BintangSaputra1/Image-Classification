import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Retakan Beton", page_icon="🏗️")
st.title("🏗️ Deteksi Retakan pada Beton")
st.write("Unggah gambar permukaan beton untuk mengetahui apakah terdapat retakan atau tidak.")

# 2. Tentukan Kelas (Label)
# Pastikan urutan kelas ini SAMA PERSIS dengan saat Anda melakukan training
class_names = ['Retak', 'Tidak_Retak']

# 3. Memuat (Load) Model dengan Caching
# @st.cache_resource digunakan agar model tidak dimuat ulang setiap kali pengguna mengunggah gambar baru
@st.cache_resource
def load_model():
    # Pastikan file model_crack_beton.h5 berada di folder yang sama dengan file streamlit_app.py ini
    model_path = 'model_crack_beton.h5' 
    return tf.keras.models.load_model(model_path)

with st.spinner("Sedang memuat model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file 'model_crack_beton.h5' ada di direktori yang sama. Error: {e}")
        model = None

# 4. Fungsi untuk Memprediksi Gambar
def prediksi_gambar_beton(image_upload, model):
    # Membaca gambar menggunakan PIL
    img = Image.open(image_upload)
    
    # Menampilkan gambar yang diunggah di web
    st.image(img, caption='Gambar yang diunggah', use_container_width=True)
    
    # Mengubah ukuran gambar sesuai target training (150x150)
    img_resized = img.resize((150, 150))
    
    # Mengubah gambar menjadi array numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
    # Menambahkan dimensi batch (menjadi [1, 150, 150, 3])
    img_array = tf.expand_dims(img_array, 0)
    
    # Melakukan prediksi
    with st.spinner("Menganalisis gambar..."):
        predictions = model.predict(img_array)
        
    # Logika prediksi sesuai dengan kode Colab Anda
    score = tf.nn.softmax(predictions[0]) if len(class_names) > 2 else predictions[0]
    
    # Cek jika model menggunakan klasifikasi biner dengan sigmoid (output 1 neuron)
    if len(class_names) == 2 and predictions.shape[-1] == 1:
        predicted_class_idx = 1 if predictions[0][0] >= 0.5 else 0
        konfidensi = predictions[0][0] if predicted_class_idx == 1 else 1 - predictions[0][0]
        konfidensi = konfidensi * 100
    else:
        # Jika menggunakan softmax atau output multi-kolom
        predicted_class_idx = np.argmax(predictions[0])
        konfidensi = np.max(score) * 100
        
    hasil_prediksi = class_names[predicted_class_idx]
    
    # Menampilkan hasil dengan warna yang sesuai
    st.subheader("Hasil Prediksi:")
    if hasil_prediksi == 'Retak':
        st.error(f"⚠️ **{hasil_prediksi}** (Tingkat Keyakinan: {konfidensi:.2f}%)")
    else:
        st.success(f"✅ **{hasil_prediksi}** (Tingkat Keyakinan: {konfidensi:.2f}%)")

# 5. Antarmuka Upload Gambar
if model is not None:
    uploaded_file = st.file_uploader("Pilih gambar beton...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.divider()
        # Jalankan prediksi jika ada file yang diunggah
        prediksi_gambar_beton(uploaded_file, model)
