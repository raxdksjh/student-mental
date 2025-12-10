import joblib
import pandas as pd

# Load model yang sudah dibuat di model.py
model = joblib.load('mental_health_model.pkl')

def get_prediction(data_input):
    """
    Menerima input dictionary dari app.py, 
    mengubahnya jadi format angka, 
    dan mengembalikan hasil prediksi.
    """
    
    # 1. Bikin DataFrame dari input
    df = pd.DataFrame(data_input, index=[0])
    
    # 2. Encoding Manual (Mengubah Teks User jadi Angka buat Model)
    # Ini harus sama persis urutannya dengan model.py tadi
    
    # Mapping Course
    course_map = {
        'Computer Science': 0, 
        'Engineering': 1, 
        'Business': 2, 
        'Law': 3, 
        'Medical': 4, 
        'Others': 5
    }
    
    # Mapping Gender
    gender_map = {'Female': 0, 'Male': 1}
    
    # Mapping Social Support
    support_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    
    # Terapkan Mapping
    df['Course'] = df['Course'].map(course_map)
    df['Gender'] = df['Gender'].map(gender_map)
    df['Social_Support'] = df['Social_Support'].map(support_map)
    
    # 3. Lakukan Prediksi
    prediction = model.predict(df)
    
    # 4. Kembalikan Hasil (0 atau 1)
    if prediction[0] == 1:
        return "STRES (High Risk)"
    else:
        return "NORMAL (Low Risk)"