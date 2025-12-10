import streamlit as st
import prediction as pred  # Memanggil file prediction.py kita tadi

# Judul Halaman
st.set_page_config(page_title="Mental Health Prediction", page_icon="🧠")
st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")
st.divider()

# --- SIDEBAR INPUT ---
st.sidebar.header("📝 Masukkan Data")

# Input 1: Jurusan (Sesuai Feature Importance kamu)
course_input = st.sidebar.selectbox(
    "Jurusan (Course)",
    ('Computer Science', 'Engineering', 'Business', 'Law', 'Medical', 'Others')
)

# Input 2: IPK (Sesuai kolom CGPA dataset kamu)
cgpa_input = st.sidebar.slider(
    "IPK (CGPA)", 0.0, 4.0, 3.5
)

# Input 3: Gender
gender_input = st.sidebar.radio(
    "Jenis Kelamin", ('Female', 'Male')
)

# Input 4: Social Support (Sesuai grafik bar chart kamu)
support_input = st.sidebar.select_slider(
    "Dukungan Sosial", 
    options=['Low', 'Moderate', 'High']
)

# Input 5: Usia (Sesuai kolom Age)
age_input = st.sidebar.number_input(
    "Usia (Age)", 17, 30, 21
)

# --- TOMBOL PREDIKSI ---
if st.button("🚀 Prediksi Sekarang"):
    
    # Bungkus data jadi dictionary biar rapi dikirim ke prediction.py
    user_data = {
        'Course': course_input,
        'CGPA': cgpa_input,
        'Gender': gender_input,
        'Social_Support': support_input,
        'Age': age_input
    }
    
    # Panggil fungsi dari file prediction.py
    hasil = pred.get_prediction(user_data)
    
    # Tampilkan Hasil
    if "STRES" in hasil:
        st.error(f"⚠️ Hasil Prediksi: **{hasil}**")
        st.write("Berdasarkan pola data, mahasiswa dengan profil ini memiliki risiko tinggi mengalami tekanan mental.")
    else:
        st.success(f"✅ Hasil Prediksi: **{hasil}**")
        st.write("Kondisi mahasiswa diprediksi stabil.")

else:
    st.info("Silakan masukkan data di sidebar dan tekan tombol prediksi.")