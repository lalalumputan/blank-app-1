
import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

# -----------------------------
# 1. Load Data
# -----------------------------
sample_data = pd.read_csv("student_data_typed.csv")

# Definisi fitur (tanpa previous_gpa)
categorical_features = ['access_to_tutoring', 'dropout_risk', 'study_environment']
numerical_features = ['motivation_level', 'study_hours_per_day', 'exam_anxiety_score', 'screen_time']

X = sample_data[categorical_features + numerical_features]
y = sample_data["exam_score"]

# -----------------------------
# 2. Preprocessing dan Model
# -----------------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=5, random_state=42))
])

model.fit(X, y)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🎓 Prediksi Nilai Ujian Mahasiswa (Tanpa GPA)")

st.write("Masukkan informasi mahasiswa berikut:")

# Input user
tutoring = st.selectbox("Akses ke Bimbingan Belajar", sample_data['access_to_tutoring'].unique())
dropout = st.selectbox("Risiko Dropout", sample_data['dropout_risk'].unique())
environment = st.selectbox("Lingkungan Belajar", sample_data['study_environment'].unique())
motivation = st.slider("Tingkat Motivasi", 0.0, 10.0, 5.0, 0.1)
hours = st.slider("Jam Belajar per Hari", 0.0, 10.0, 2.0, 0.5)
anxiety = st.slider("Skor Kecemasan Ujian", 0.0, 10.0, 5.0, 0.1)
screen = st.slider("Waktu Layar (jam/hari)", 0.0, 10.0, 2.0, 0.5)

# Masukkan ke dalam DataFrame
input_df = pd.DataFrame([{
    'access_to_tutoring': tutoring,
    'dropout_risk': dropout,
    'study_environment': environment,
    'motivation_level': motivation,
    'study_hours_per_day': hours,
    'exam_anxiety_score': anxiety,
    'screen_time': screen
}])

# Prediksi
prediction = model.predict(input_df)[0]

# -----------------------------
# 4. Output
# -----------------------------
st.subheader("📋 Data Input")
st.write(input_df)

st.success(f"📘 Prediksi Nilai Ujian Mahasiswa: **{prediction:.2f}**")
