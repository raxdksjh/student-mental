import joblib
import pandas as pd
import os

# =========================================
# BAGIAN PERBAIKAN PATH (ALAMAT FILE)
# =========================================

# 1. Cari tahu alamat lengkap di mana file 'prediction.py' ini disimpan
# (Contoh hasil: /mount/src/repo-kamu/mental/)
current_directory = os.path.dirname(os.path.abspath(__file__))

# 2. Gabungkan alamat folder tersebut dengan nama file model
# (Contoh hasil: /mount/src/repo-kamu/mental/mental_health_model.pkl)
model_path = os.path.join(current_directory, 'mental_health_model.pkl')

# 3. Load model menggunakan alamat lengkap (Absolute Path)
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    # Pesan error khusus biar kita tau kalau masih gagal
    raise FileNotFoundError(f"❌ File tidak ditemukan di alamat: {model_path}. Pastikan file .pkl sudah di-upload ke GitHub!")

# =========================================
# FUNGSI PREDIKSI
# =========================================

def get_prediction(data_input):
    # 1. Bikin DataFrame dari input user
    df = pd.DataFrame(data_input, index=[0])
    
    # 2. Mapping Manual (Mengubah Teks jadi Angka)
    # Ini HARUS SAMA PERSIS urutannya dengan waktu training di model.py
    
    course_map = {
        'Computer Science': 0, 
        'Engineering': 1, 
        'Business': 2, 
        'Law': 3, 
        'Medical': 4, 
        'Others': 5
    }
    
    gender_map = {'Female': 0, 'Male': 1}
    
    support_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    
    # Terapkan Mapping
    df['Course'] = df['Course'].map(course_map)
    df['Gender'] = df['Gender'].map(gender_map)
    df['Social_Support'] = df['Social_Support'].map(support_map)
    
    # 3. Lakukan Prediksi dengan Model
    prediction = model.predict(df)
    
    # 4. Kembalikan Hasil
    if prediction[0] == 1:
        return "STRES (High Risk)"
    else:
        return "NORMAL (Low Risk)"
