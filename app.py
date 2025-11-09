import streamlit as st
import joblib
import librosa
import numpy as np
import io
from st_audiorec import st_audiorec  # Library untuk merekam audio

# --- Konfigurasi Awal (dari Notebook) ---
SR = 16000
DURATION = 1
SAMPLES = SR * DURATION
THRESHOLD = 85  # Threshold keyakinan (80%)

# --- Fungsi untuk Memuat Model (dengan Caching) ---
@st.cache_resource
def load_models():
    """Memuat model, scaler, dan label encoder yang telah disimpan."""
    try:
        model = joblib.load("voice_model.pkl")
        scaler = joblib.load("voice_scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except FileNotFoundError:
        st.error("Error: File model tidak ditemukan.")
        st.info("Pastikan file 'voice_model.pkl', 'voice_scaler.pkl', dan 'label_encoder.pkl' berada di folder yang sama dengan app.py")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None

# --- Fungsi Ekstraksi Fitur (dari Notebook) ---
def extract_mfcc(y, sr=SR):
    """
    Ekstrak fitur MFCC dari sinyal audio.
    Audio akan dipotong atau diberi padding agar pas 1 detik.
    """
    # Crop / padding
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, max(0, SAMPLES - len(y))), "constant")
    
    # Ekstraksi MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# --- Fungsi untuk Prediksi ---
def predict_audio(y, sr, model, scaler, le):
    """Melakukan prediksi pada data audio yang diberikan."""
    # 1. Ekstraksi fitur
    feature = extract_mfcc(y=y, sr=sr)
    
    # 2. Scaling
    # Reshape karena scaler mengharapkan input 2D
    feature_scaled = scaler.transform([feature])
    
    # 3. Prediksi
    pred_index = model.predict(feature_scaled)[0]
    pred_label = le.inverse_transform([pred_index])[0]
    
    # 4. Confidence (Probabilitas)
    prob = model.predict_proba(feature_scaled)[0]
    confidence = max(prob) * 100
    
    return pred_label, confidence

# --- Membangun UI Streamlit ---

st.set_page_config(page_title="Prediksi Perintah Suara", layout="centered")
st.title("🎙️ Aplikasi Prediksi Perintah Suara")
st.write("""
Aplikasi ini menggunakan model Support Vector Machine (SVM) untuk mengenali 
perintah suara ('buka' atau 'tutup') dari dua orang spesifik 
yang terdaftar dalam dataset.
""")

# Memuat model
model, scaler, le = load_models()

# Hanya tampilkan UI jika model berhasil dimuat
if model is not None and scaler is not None and le is not None:
    
    st.sidebar.info(f"**Label Terdaftar:**\n- {le.classes_[0]}\n- {le.classes_[1]}\n- {le.classes_[2]}\n- {le.classes_[3]}")
    st.sidebar.info(f"**Threshold Keyakinan:** {THRESHOLD}%")

    # --- Opsi 1: Upload File ---
    st.header("Opsi 1: Unggah File Audio (.wav)")
    uploaded_file = st.file_uploader("Pilih file .wav...", type=["wav"])

    if uploaded_file is not None:
        try:
            # Tampilkan audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Muat audio dengan librosa
            # Gunakan io.BytesIO untuk membaca file yang diupload
            y, sr_up = librosa.load(io.BytesIO(uploaded_file.read()), sr=SR)
            
            # Lakukan prediksi
            label, confidence = predict_audio(y, sr_up, model, scaler, le)
            
            # Tampilkan hasil
            st.subheader("Hasil Prediksi (dari File):")
            if confidence >= THRESHOLD:
                st.success(f"✅ Dikenali sebagai: **{label}**")
                st.info(f"Keyakinan: **{confidence:.2f}%**")
            else:
                st.error(f"❌ Suara tidak dikenali.")
                st.info(f"Prediksi terdekat: {label} (Keyakinan: {confidence:.2f}%) - Di bawah threshold {THRESHOLD}%")
        
        except Exception as e:
            st.error(f"Error saat memproses file audio: {e}")

    st.divider()

    # --- Opsi 2: Rekam Suara Langsung ---
    st.header("Opsi 2: Rekam Suara Langsung")
    st.info("Klik ikon mikrofon untuk merekam. Usahakan durasi rekaman sekitar 1 detik.")

    # Widget perekam audio
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        try:
            # Tampilkan audio player dari rekaman
            st.audio(wav_audio_data, format='audio/wav')
            
            # Muat audio dari rekaman (yang berupa bytes)
            y_rec, sr_rec = librosa.load(io.BytesIO(wav_audio_data), sr=SR)
            
            # Lakukan prediksi
            label_rec, confidence_rec = predict_audio(y_rec, sr_rec, model, scaler, le)
            
            # Tampilkan hasil
            st.subheader("Hasil Prediksi (dari File):")
            if confidence >= THRESHOLD:
                st.success(f"✅ Dikenali sebagai: **{label}**")
                st.info(f"Keyakinan: **{confidence:.2f}%**")
            else:
                # --- MODIFIKASI DI SINI ---
                # Ambil hanya perintah (misal: "tutup" dari "tutup_orang_1")
                perintah_terdekat = label.split('_')[0]
                
                st.error(f"❌ Suara tidak dikenali.")
                st.info(f"Prediksi terdekat: **{perintah_terdekat}** (Keyakinan: {confidence:.2f}%) - Di bawah threshold {THRESHOLD}%")

        except Exception as e:
            st.error(f"Error saat memproses audio rekaman: {e}")

else:
    st.error("Aplikasi tidak dapat berjalan karena model gagal dimuat.")
