import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Bikin Data Dummy (Representasi dari Project Kamu)
# Kita bikin pola sederhana: CS + Low Support = Stres (1)
data = {
    'Course':           [0, 0, 1, 1, 2, 2, 3, 4, 5, 0], # 0:CS, 1:Engineering, dst
    'CGPA':             [3.8, 2.5, 3.9, 2.6, 3.5, 3.2, 3.0, 3.8, 3.5, 2.4],
    'Gender':           [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # 0:Female, 1:Male
    'Social_Support':   [2, 0, 2, 0, 1, 1, 0, 2, 1, 0], # 0:Low, 1:Mod, 2:High
    'Age':              [20, 21, 19, 22, 20, 23, 21, 20, 22, 21],
    'Stress_Level':     [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]  # Target: 0 (Normal), 1 (Stres)
}

df = pd.DataFrame(data)

# Keterangan Mapping (Biar kamu tau urutannya):
# Course: 0:Computer Science, 1:Engineering, 2:Business, 3:Law, 4:Medical, 5:Others
# Gender: 0:Female, 1:Male
# Social_Support: 0:Low, 1:Moderate, 2:High

# 2. Pisahkan Fitur (X) dan Target (y)
X = df[['Course', 'CGPA', 'Gender', 'Social_Support', 'Age']]
y = df['Stress_Level']

# 3. Latih Model Sederhana
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 4. Simpan Model jadi File .pkl
joblib.dump(model, 'mental_health_model.pkl')

print("✅ Model berhasil dibuat dan disimpan sebagai 'mental_health_model.pkl'")