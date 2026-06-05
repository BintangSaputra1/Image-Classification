import os
# WAJIB DI BARIS PALING ATAS: Memaksa penggunaan Keras versi lama (Keras 2)
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import streamlit as st
import tf_keras as tfk
import numpy as np
from PIL import Image
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
        # Menggunakan parameter 'id' langsung di gdown lebih stabil untuk melewati peringatan virus Google Drive
        file_id = '1yaUHZ5p6aSxFuRYduiQKMwWpIJwf-if3' 
        try:
            gdown.download(id=file_id, output=model_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh model dari Google Drive: {e}")
            return None
        
    # Memuat model
    try:
        model = tfk.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file tidak corrupt dan requirements.txt sudah benar. Detail: {e}")
        return None

# 4. Fungsi untuk Memprediksi Gambar
def prediksi_gambar(image_pil, model):
    img_height = 150
    img_width = 150

    # Ubah ukuran gambar menggunakan library PIL (Pillow)
    img_resized = image_pil.resize((img_width, img_height))

    # Ubah gambar menjadi array numpy secara langsung (menghindari error fungsi preprocessing tfk)
    img_array = np.array(img_resized, dtype=np.float32)

    # Tambahkan dimensi batch: (150, 150, 3) menjadi (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Opsional: Jika modelmu dilatih dengan normalisasi (/255.0), aktifkan baris di bawah ini:
    # img_array = img_array / 255.0

    # Melakukan prediksi
    predictions = model.predict(img_array)
    score = predictions[0]

    # Logika persentase 
    if len(class_names) == 2 and predictions.shape[-1] == 1:
        # Jika menggunakan fungsi aktivasi Sigmoid
        predicted_class_idx = 1 if score[0] >= 0.5 else 0
        konfidensi = score[0] if predicted_class_idx == 1 else 1 - score[0]
        konfidensi = konfidensi * 100
    else:
        # Jika menggunakan fungsi aktivasi Softmax
        predicted_class_idx = np.argmax(score)
        konfidensi = np.max(score) * 100

    hasil_prediksi = class_names[predicted_class_idx]
    return hasil_prediksi, konfidensi

# 5. Antarmuka Pengguna (UI) Streamlit
st.title("🏗️ Deteksi Retak pada Beton")
st.write("Unggah foto permukaan beton untuk mendeteksi apakah terdapat retakan atau tidak menggunakan model Artificial Intelligence.")

# Memuat model di latar belakang
with st.spinner("Sedang menyiapkan model AI... (Ini mungkin memakan waktu beberapa saat jika harus mengunduh)"):
    model_beton = load_model()

# Menghentikan eksekusi jika model gagal dimuat
if model_beton is None:
    st.stop()

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
