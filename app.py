import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Well-being Analysis", 
    page_icon="📊",
    layout="wide"
)

# CSS personnalisé - style minimal
st.markdown("""
<style>
    .stApp {
        background-color: #F5F5F5;
    }
    
    .main-header {
        text-align: center;
        padding: 1rem;
        background-color: #4A90E2;
        border-radius: 5px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
    
    .stButton > button {
        background-color: #4A90E2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border-radius: 3px;
    }
    
    .stButton > button:hover {
        background-color: #3A7BC8;
    }
    
    .stButton > button:disabled {
        background-color: #BDBDBD;
    }
    
    .section-complete {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 3px;
    }
    
    .section-incomplete {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("pipeline.pkl")

# Header
st.markdown('<div class="main-header"><h1>Student Well-being Analysis</h1></div>', unsafe_allow_html=True)

# État des sections
if 'section1_complete' not in st.session_state:
    st.session_state.section1_complete = False
if 'section2_complete' not in st.session_state:
    st.session_state.section2_complete = False
if 'section3_complete' not in st.session_state:
    st.session_state.section3_complete = False

# Indicateur de progression
st.markdown("### Formulaire")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.section1_complete:
        st.markdown('<div class="section-complete">✓ Section 1 : Profil</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-incomplete">○ Section 1 : Profil</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.section2_complete:
        st.markdown('<div class="section-complete">✓ Section 2 : Academique</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-incomplete">○ Section 2 : Academique</div>', unsafe_allow_html=True)

with col3:
    if st.session_state.section3_complete:
        st.markdown('<div class="section-complete">✓ Section 3 : Lifestyle</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-incomplete">○ Section 3 : Lifestyle</div>', unsafe_allow_html=True)

st.markdown("---")

# Section 1
with st.expander("Section 1 : Profil personnel", expanded=not st.session_state.section1_complete):
    c1, c2 = st.columns(2)
    age = c1.slider("Age", 18, 60, 22)
    gender = c2.selectbox("Genre", ["Homme", "Femme"])
    city = st.selectbox("Ville", ["Delhi", "Mumbai", "Bangalore", "Autre"])
    family_history = st.radio("Antecedents familiaux de troubles mentaux ?", ["Non", "Oui"], horizontal=True)
    
    if st.button("Valider section 1", key="btn1"):
        st.session_state.section1_complete = True
        st.rerun()
    
    if st.session_state.section1_complete:
        if st.button("Modifier section 1", key="edit1"):
            st.session_state.section1_complete = False
            st.rerun()

# Section 2
with st.expander("Section 2 : Situation academique", expanded=not st.session_state.section2_complete):
    c3, c4 = st.columns(2)
    academic_pressure = c3.select_slider("Pression academique", options=[1, 2, 3, 4, 5], value=3)
    financial_stress = c4.select_slider("Stress financier", options=[1, 2, 3, 4, 5], value=2)
    grade_20 = st.number_input("Moyenne generale (/20)", 0.0, 20.0, 14.0)
    study_sat = st.select_slider("Satisfaction des etudes", options=[1, 2, 3, 4, 5], value=3)
    degree = st.selectbox("Diplome", ["Licence", "Master", "Doctorat", "Autre"])
    
    if st.button("Valider section 2", key="btn2"):
        st.session_state.section2_complete = True
        st.rerun()
    
    if st.session_state.section2_complete:
        if st.button("Modifier section 2", key="edit2"):
            st.session_state.section2_complete = False
            st.rerun()

# Section 3
with st.expander("Section 3 : Style de vie", expanded=not st.session_state.section3_complete):
    c5, c6 = st.columns(2)
    sleep = c5.slider("Duree de sommeil (heures)", 3.0, 12.0, 7.0)
    work_hours = c6.number_input("Heures d'etude/travail par jour", 0, 15, 6)
    diet = st.selectbox("Habitudes alimentaires", ["Sain", "Modere", "Malsain"], index=1)
    suicidal = st.selectbox("Pensees negatives recurrentes ?", ["Non", "Parfois", "Souvent"])
    
    if st.button("Valider section 3", key="btn3"):
        st.session_state.section3_complete = True
        st.rerun()
    
    if st.session_state.section3_complete:
        if st.button("Modifier section 3", key="edit3"):
            st.session_state.section3_complete = False
            st.rerun()

st.markdown("---")

# Bouton d'analyse
all_sections_complete = st.session_state.section1_complete and st.session_state.section2_complete and st.session_state.section3_complete

if all_sections_complete:
    if st.button("Lancer l'analyse", use_container_width=True):
        # Préparation des données
        data = pd.DataFrame([{
            'Gender': 'Male' if gender == "Homme" else 'Female',
            'Age': age,
            'City': city,
            'Profession': "Student",
            'Academic Pressure': academic_pressure,
            'Work Pressure': 0,
            'CGPA': grade_20 / 2,
            'Study Satisfaction': study_sat,
            'Job Satisfaction': 0,
            'Sleep Duration': sleep,
            'Dietary Habits': 'Healthy' if diet == "Sain" else 'Moderate' if diet == "Modere" else 'Unhealthy',
            'Degree': degree,
            'suicidal_thoughts': 'Yes' if suicidal in ["Parfois", "Souvent"] else 'No',
            'Work/Study Hours': work_hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }])
        
        with st.spinner("Analyse en cours..."):
            try:
                model = load_model()
                proba = model.predict_proba(data)[0]
                risk_score = proba[1]
                well_being_score = (1 - risk_score) * 100
                
                # Affichage des résultats
                st.markdown("---")
                st.markdown("### Resultats de l'analyse")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    
                    # Barre de progression
                    st.markdown(f"""
                    <div style="margin: 20px 0;">
                        <div style="background-color: #E0E0E0; border-radius: 3px; height: 25px; overflow: hidden;">
                            <div style="background-color: #4A90E2; width: {well_being_score}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; font-weight: bold;">
                            Score: {well_being_score:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Interprétation simple
                st.markdown("---")
                st.markdown("### Interpretation")
                
                if well_being_score < 40:
                    st.error("Niveau de stress eleve. Une attention particuliere est recommandee.")
                elif well_being_score < 70:
                    st.warning("Niveau de stress modere. Quelques ajustements pourraient etre benefiques.")
                else:
                    st.success("Bon niveau de bien-etre. Continuez ainsi !")
                
                # Bouton nouvelle analyse
                st.markdown("---")
                if st.button("Nouvelle analyse", use_container_width=True):
                    # Réinitialisation
                    st.session_state.section1_complete = False
                    st.session_state.section2_complete = False
                    st.session_state.section3_complete = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {str(e)}")
else:
    st.info("Veuillez valider les 3 sections pour lancer l'analyse.")