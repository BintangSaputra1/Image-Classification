import streamlit as st
import tf_keras as tfk  # Menggunakan tf-keras secara langsung, bukan tensorflow!
import numpy as np
from PIL import Image
import os
import gdown

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Retak Beton", layout="centered", page_icon="🏗️")

# 2. Definisikan Kelas (Label)
class_names = ['Retak', 'Tidak_Retak']

# 3. Fungsi untuk Memuat Model dengan Caching & Download Otomatis
@st.cache_resource
def load_model():
    model_path = 'model_crack_beton.h5'
    
    # Mengecek apakah file model sudah ada di server
    if not os.path.exists(model_path):
        # Download dari Google Drive menggunakan ID Anda
        file_id = '1yaUHZ5p6aSxFuRYduiQKMwWpIJwf-if3' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    # Memuat model menggunakan tf_keras (Keras 2)
    return tfk.models.load_model(model_path, compile=False)

# 4. Fungsi untuk Memprediksi Gambar
def prediksi_gambar(image_pil, model):
    img_height = 150
    img_width = 150

    # Ubah ukuran gambar menggunakan library PIL (Pillow)
    img_resized = image_pil.resize((img_width, img_height))

    # Ubah gambar menjadi array numpy menggunakan tfk
    img_array = tfk.preprocessing.image.img_to_array(img_resized)

    # Tambahkan dimensi batch menggunakan numpy (menggantikan tf.expand_dims)
    img_array = np.expand_dims(img_array, axis=0)

    # Melakukan prediksi
    predictions = model.predict(img_array)

    # Logika persentase (Disesuaikan dengan numpy)
    score = predictions[0]

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
with st.spinner("Sedang mengunduh dan memuat model AI... (Ini mungkin memakan waktu beberapa saat)"):
    model_beton = load_model()

# 6. Fitur Upload Gambar (Uploader File)
uploaded_file = st.file_uploader("Pilih gambar beton Anda...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    
    if st.button("Deteksi Gambar"):
        with st.spinner("Sedang menganalisis gambar..."):
            hasil, konfidensi = prediksi_gambar(image, model_beton)
        
        if hasil == 'Retak':
            st.error(f"⚠️ **Hasil Prediksi:** Beton Terdeteksi **{hasil}**")
            st.write(f"**Tingkat Keyakinan Model:** {konfidensi:.2f}%")
        else:
            st.success(f"✅ **Hasil Prediksi:** Beton Terdeteksi **{hasil.replace('_', ' ')}**")
            st.write(f"**Tingkat Keyakinan Model:** {konfidensi:.2f}%")
