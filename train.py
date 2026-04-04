import streamlit as st
import pandas as pd
import joblib

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Well-being Analysis", layout="centered")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("pipeline.pkl")

model = load_model()

# ==============================
# TITLE
# ==============================
st.title("🧠 Student Well-being Analysis")
st.markdown("This tool provides an estimation of well-being based on lifestyle and academic factors.")

# ==============================
# INPUTS
# ==============================

# --- Profil ---
age = st.slider("Age", 18, 60, 22)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Other"])
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

# --- Académique ---
academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
financial_stress = st.slider("Financial Stress", 1, 5, 2)
grade_20 = st.number_input("Average grade (/20)", 0.0, 20.0, 14.0)
study_sat = st.slider("Study Satisfaction", 1, 5, 3)
degree = st.selectbox("Degree", [
    "BSc",
    "BTech",
    "BCA",
    "B.Pharm",
    "Other"
])

# --- Lifestyle ---
sleep = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0)
work_hours = st.slider("Work/Study Hours per day", 0, 15, 6)
diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
suicidal = st.selectbox("Negative thoughts frequency", ["No", "Sometimes", "Often"])

# ==============================
# PREDICTION
# ==============================

if st.button("Analyze Well-being"):

    # 🔄 Conversion CGPA (cohérent avec train)
    cgpa = grade_20 / 2

    # 🔄 Simplification cohérente
    suicidal_value = "Yes" if suicidal in ["Sometimes", "Often"] else "No"

    # ==============================
    # DATAFRAME (⚠️ EXACT COLONNES)
    # ==============================
    data = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'City': city,
        'Profession': "Student",
        'Academic Pressure': academic_pressure,
        'Work Pressure': 0,
        'CGPA': cgpa,
        'Study Satisfaction': study_sat,
        'Job Satisfaction': 0,
        'Sleep Duration': sleep,
        'Dietary Habits': diet,
        'Degree': degree,
        'suicidal_thoughts': suicidal_value,
        'Work/Study Hours': work_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }])

    # ==============================
    # PREDICTION
    # ==============================
    proba = model.predict_proba(data)[0]
    risk_score = proba[1]
    well_being_score = (1 - risk_score) * 100

    # ==============================
    # RESULT DISPLAY
    # ==============================
    st.subheader("📊 Result")

    st.progress(int(well_being_score))

    st.write(f"### Well-being Score: **{well_being_score:.1f}%**")

    # Interpretation douce (moins stressante)
    if well_being_score < 40:
        st.warning("Your current state suggests some difficulties. It may help to talk to someone or adjust certain habits.")
    elif well_being_score < 70:
        st.info("Your situation is moderate. Small improvements in lifestyle or workload balance could help.")
    else:
        st.success("Your well-being level looks good. Keep maintaining your healthy habits!")

    # ==============================
    # DISCLAIMER
    # ==============================
    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and does not replace professional medical advice.")