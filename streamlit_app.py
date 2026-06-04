import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman 
st.set_page_config(page_title="Prediksi Retak Beton", layout="centered", page_icon="🏗️")

# 2. Definisikan Kelas (Label)
# Pastikan urutan kelas ini SAMA PERSIS dengan saat Anda melakukan training
class_names = ['Retak', 'Tidak_Retak']

# 3. Fungsi untuk Memuat Model dengan Caching
# @st.cache_resource sangat penting agar model tidak diload ulang setiap kali user mengklik sesuatu di layar
@st.cache_resource
def load_model():
    # Pastikan file model ('model_crack_beton.h5') diletakkan satu folder dengan file 'streamlit_app.py' ini.
    # Jika diletakkan di folder lain, ubah path-nya.
    return tf.keras.models.load_model('model_crack_beton.h5')

# 4. Fungsi untuk Memprediksi Gambar
def prediksi_gambar(image_pil, model):
    img_height = 150
    img_width = 150

    # Ubah ukuran gambar menggunakan library PIL (Pillow)
    img_resized = image_pil.resize((img_width, img_height))

    # Ubah gambar menjadi array numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)

    # Tambahkan dimensi batch (dari [150, 150, 3] menjadi [1, 150, 150, 3])
    img_array = tf.expand_dims(img_array, 0)

    # Melakukan prediksi
    predictions = model.predict(img_array)

    # Logika persentase (Diambil persis dari logika Colab Anda)
    score = tf.nn.softmax(predictions[0]) if len(class_names) > 2 else predictions[0]

    if len(class_names) == 2 and predictions.shape[-1] == 1:
        # Jika menggunakan fungsi aktivasi Sigmoid
        predicted_class_idx = 1 if predictions[0][0] >= 0.5 else 0
        konfidensi = predictions[0][0] if predicted_class_idx == 1 else 1 - predictions[0][0]
        konfidensi = konfidensi * 100
    else:
        # Jika menggunakan fungsi aktivasi Softmax
        predicted_class_idx = np.argmax(predictions[0])
        konfidensi = np.max(score) * 100

    hasil_prediksi = class_names[predicted_class_idx]
    return hasil_prediksi, konfidensi

# 5. Antarmuka Pengguna (UI) Streamlit
st.title("🏗️ Deteksi Retak pada Beton")
st.write("Unggah foto permukaan beton untuk mendeteksi apakah terdapat retakan atau tidak menggunakan model Artificial Intelligence.")

# Memuat model di latar belakang
with st.spinner("Sedang memuat model AI... (Ini mungkin memakan waktu sebentar)"):
    model_beton = load_model()

# 6. Fitur Upload Gambar (Uploader File)
uploaded_file = st.file_uploader("Pilih gambar beton Anda...", type=["jpg", "jpeg", "png"])

# Jika pengguna sudah mengunggah file
if uploaded_file is not None:
    # Buka gambar menggunakan PIL dan pastikan formatnya RGB
    image = Image.open(uploaded_file).convert('RGB')
    
    # Menampilkan gambar yang diunggah
    st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    
    # Tombol Prediksi
    if st.button("Deteksi Gambar"):
        with st.spinner("Sedang menganalisis gambar..."):
            # Panggil fungsi prediksi
            hasil, konfidensi = prediksi_gambar(image, model_beton)
        
        # Menampilkan hasil dengan warna yang berbeda tergantung prediksi
        if hasil == 'Retak':
            st.error(f"⚠️ **Hasil Prediksi:** Beton Terdeteksi **{hasil}**")
            st.write(f"**Tingkat Keyakinan Model:** {konfidensi:.2f}%")
        else:
            st.success(f"✅ **Hasil Prediksi:** Beton Terdeteksi **{hasil.replace('_', ' ')}**")
            st.write(f"**Tingkat Keyakinan Model:** {konfidensi:.2f}%")
