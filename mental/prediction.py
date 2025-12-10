import joblib
import pandas as pd
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_directory, 'mental_health_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ File tidak ditemukan di alamat: {model_path}. Pastikan file .pkl sudah di-upload ke GitHub!")


def get_prediction(data_input):
    # 1. Bikin DataFrame dari input user
    df = pd.DataFrame(data_input, index=[0])
    
    # 2. Mapping Manual (Mengubah Teks jadi Angka)
    
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

