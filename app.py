import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Well-being Analysis", 
    page_icon="📊",
    layout="wide"
)

# CSS personnalisé
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
    """Charge le modèle et les encodeurs"""
    try:
        model = joblib.load("pipeline.pkl")
        encoders = joblib.load("encoders.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, encoders, feature_columns
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None, None, None

# ─────────────────────────────────────────────
# CORRECTIF 1 : mappings FR → EN pour les encodeurs
# Ces valeurs doivent correspondre exactement à ce que
# le LabelEncoder a vu pendant l'entraînement.
# ─────────────────────────────────────────────
GENDER_MAP = {
    "Homme": "Male",
    "Femme": "Female"
}

DIET_MAP = {
    "Sain":    "Healthy",
    "Modéré":  "Moderate",
    "Malsain": "Unhealthy"
}

FAMILY_MAP = {
    "Non": "No",
    "Oui": "Yes"
}

SUICIDAL_MAP = {
    "Non":     "No",
    "Parfois": "Yes",
    "Souvent": "Yes"
}

# Degree : utilise les valeurs réelles du dataset
DEGREE_OPTIONS = ["B.Ed", "B.Tech", "BA", "BCA", "BHM", "BPharm", "BSc",
                  "Class 12", "LLB", "M.Ed", "M.Tech", "MA", "MBA", "MCA",
                  "MPharm", "MSc", "PhD"]

# Villes présentes dans le dataset
CITY_OPTIONS = ["Agra", "Ahmedabad", "Bangalore", "Bhopal", "Chennai",
                "Delhi", "Faridabad", "Ghaziabad", "Hyderabad", "Jaipur",
                "Kalyan", "Kolkata", "Lucknow", "Ludhiana", "Mumbai",
                "Nagpur", "Patna", "Pune", "Rajkot", "Srinagar",
                "Surat", "Thane", "Vadodara", "Vasai-Virar", "Visakhapatnam"]

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
    # CORRECTIF 2 : villes issues du dataset réel
    city = st.selectbox("Ville", CITY_OPTIONS, index=CITY_OPTIONS.index("Delhi"))
    family_history = st.radio(
        "Antécédents familiaux de troubles mentaux ?",
        ["Non", "Oui"],
        horizontal=True
    )
    
    if st.button("Valider section 1", key="btn1"):
        st.session_state.section1_complete = True
        st.rerun()
    
    if st.session_state.section1_complete:
        if st.button("Modifier section 1", key="edit1"):
            st.session_state.section1_complete = False
            st.rerun()

# Section 2
with st.expander("Section 2 : Situation académique", expanded=not st.session_state.section2_complete):
    c3, c4 = st.columns(2)
    academic_pressure = c3.select_slider("Pression académique", options=[1, 2, 3, 4, 5], value=3)
    financial_stress  = c4.select_slider("Stress financier",    options=[1, 2, 3, 4, 5], value=2)
    grade_20 = st.number_input("Moyenne générale (/20)", 0.0, 20.0, 14.0)
    study_sat = c3.select_slider("Satisfaction des études", options=[1, 2, 3, 4, 5], value=3)
    # CORRECTIF 3 : diplômes issus du dataset réel
    degree = st.selectbox("Diplôme", DEGREE_OPTIONS, index=DEGREE_OPTIONS.index("BSc"))
    
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

    # CORRECTIF 4 : Sleep Duration — on utilise les buckets du dataset
    # Le modèle a été entraîné sur 4 / 5.5 / 7.5 / 9
    sleep_label = c5.selectbox(
        "Durée de sommeil",
        ["Moins de 5h", "5-6h", "7-8h", "Plus de 8h"],
        index=2
    )
    SLEEP_MAP = {
        "Moins de 5h": 4.0,
        "5-6h":        5.5,
        "7-8h":        7.5,
        "Plus de 8h":  9.0
    }

    work_hours = c6.number_input("Heures d'étude/travail par jour", 0, 15, 6)
    diet = st.selectbox("Habitudes alimentaires", ["Sain", "Modéré", "Malsain"], index=1)
    suicidal = st.selectbox("Pensées négatives récurrentes ?", ["Non", "Parfois", "Souvent"])
    
    if st.button("Valider section 3", key="btn3"):
        st.session_state.section3_complete = True
        st.rerun()
    
    if st.session_state.section3_complete:
        if st.button("Modifier section 3", key="edit3"):
            st.session_state.section3_complete = False
            st.rerun()

st.markdown("---")

# Bouton d'analyse
all_sections_complete = (
    st.session_state.section1_complete and
    st.session_state.section2_complete and
    st.session_state.section3_complete
)

if all_sections_complete:
    if st.button("Lancer l'analyse", use_container_width=True):
        model, encoders, feature_columns = load_model()

        if model is None:
            st.stop()

        # ─────────────────────────────────────────────
        # CORRECTIF 5 : toutes les valeurs sont en anglais
        # (comme pendant l'entraînement) AVANT l'encodage
        # ─────────────────────────────────────────────
        input_dict = {
            'Gender':                          GENDER_MAP[gender],
            'Age':                             age,
            'City':                            city,          # déjà en anglais
            'Profession':                      'Student',
            'Academic Pressure':               academic_pressure,
            'Work Pressure':                   0,
            'CGPA':                            round(grade_20 / 2, 2),
            'Study Satisfaction':              study_sat,
            'Job Satisfaction':                0,
            'Sleep Duration':                  SLEEP_MAP[sleep_label],   # buckets numériques
            'Dietary Habits':                  DIET_MAP[diet],
            'Degree':                          degree,        # déjà en anglais
            'suicidal_thoughts':               SUICIDAL_MAP[suicidal],
            'Work/Study Hours':                work_hours,
            'Financial Stress':                financial_stress,
            'Family History of Mental Illness': FAMILY_MAP[family_history],
        }

        # Encodage des variables catégorielles
        for col, le in encoders.items():
            if col in input_dict:
                val = input_dict[col]
                known_classes = list(le.classes_)
                if val not in known_classes:
                    # Valeur inconnue : on prend la classe la plus fréquente (index 0)
                    st.warning(
                        f"Valeur « {val} » inconnue pour « {col} ». "
                        f"Valeur par défaut utilisée : « {known_classes[0]} »."
                    )
                    val = known_classes[0]
                input_dict[col] = le.transform([val])[0]

        # Création du DataFrame dans l'ordre exact des colonnes
        input_df = pd.DataFrame([input_dict])[feature_columns]

        with st.spinner("Analyse en cours..."):
            try:
                prediction  = model.predict(input_df)[0]
                proba       = model.predict_proba(input_df)[0]
                risk_score  = proba[1]
                well_being_score = (1 - risk_score) * 100

                st.markdown("---")
                st.markdown("### Résultats de l'analyse")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="margin: 20px 0;">
                        <div style="background-color: #E0E0E0; border-radius: 3px; height: 25px; overflow: hidden;">
                            <div style="background-color: #4A90E2; width: {well_being_score:.1f}%; height: 100%; border-radius: 3px;"></div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; font-weight: bold;">
                            Score de bien-être : {well_being_score:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### Interprétation")

                if well_being_score < 40:
                    st.error("⚠️ Niveau de stress élevé. Une attention particulière est recommandée.")
                elif well_being_score < 70:
                    st.warning("📊 Niveau de stress modéré. Quelques ajustements pourraient être bénéfiques.")
                else:
                    st.success("✅ Bon niveau de bien-être. Continuez ainsi !")

                with st.expander("Détails de l'analyse"):
                    st.write(f"**Risque de dépression :** {risk_score:.2%}")
                    st.write(f"**Confiance du modèle :** {max(proba):.2%}")
                    st.write(f"**Prédiction :** {'À risque' if prediction == 1 else 'Non à risque'}")

                st.markdown("---")
                if st.button("Nouvelle analyse", use_container_width=True):
                    for key in ['section1_complete', 'section2_complete', 'section3_complete']:
                        st.session_state[key] = False
                    st.rerun()

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {str(e)}")
else:
    st.info("Veuillez valider les 3 sections pour lancer l'analyse.")