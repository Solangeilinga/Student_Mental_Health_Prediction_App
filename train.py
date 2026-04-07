# ==============================
# 📦 Librairies
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
import warnings
warnings.filterwarnings("ignore")

# Vérification des versions
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {__import__('sklearn').__version__}")

# ==============================
# 1️⃣ Charger le dataset
# ==============================
df = pd.read_csv("Student Depression Dataset.csv")

# ==============================
# 2️⃣ Nettoyage
# ==============================
df = df.drop(columns=['id'], errors='ignore')
df.rename(columns={"Have you ever had suicidal thoughts ?": "suicidal_thoughts"}, inplace=True)

# Convertir Sleep Duration
def convert_sleep(x):
    if "Less than 5" in str(x):
        return 4
    elif "5-6" in str(x):
        return 5.5
    elif "7-8" in str(x):
        return 7.5
    elif "More than 8" in str(x):
        return 9
    else:
        return np.nan

df["Sleep Duration"] = df["Sleep Duration"].apply(convert_sleep)

# ==============================
# 3️⃣ Encodage
# ==============================
cat_cols = ['Gender', 'City', 'Profession', 'Dietary Habits', 'Degree',
            'suicidal_thoughts', 'Family History of Mental Illness']

# Sauvegarder les encodeurs pour l'inférence
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ==============================
# 4️⃣ Features / Target
# ==============================
X = df.drop(columns=['Depression'])
y = df['Depression']

# ==============================
# 5️⃣ Split 70/15/15
# ==============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ==============================
# 6️⃣ Pipeline Random Forest avec StandardScaler
# ==============================
pipeline_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline_rf.fit(X_train, y_train)

# Validation
y_val_pred_rf = pipeline_rf.predict(X_val)
print("\n=== Random Forest ===")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

# ==============================
# 7️⃣ GridSearchCV pour optimisation
# ==============================
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    pipeline_rf,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

# ==============================
# 8️⃣ Meilleur modèle
# ==============================
best_model = grid.best_estimator_

# Test du meilleur modèle
y_test_best = best_model.predict(X_test)
print("\n=== Optimized Random Forest ===")
print("Test Accuracy:", accuracy_score(y_test, y_test_best))
print(classification_report(y_test, y_test_best))

# ==============================
# 💾 Sauvegarde avec JOBLIB
# ==============================
joblib.dump(best_model, "pipeline.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\n✅ Modèle sauvegardé avec succès!")
print("📁 Fichiers créés: pipeline.pkl, encoders.pkl, feature_columns.pkl")