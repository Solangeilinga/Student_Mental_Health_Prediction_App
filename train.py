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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import pickle
import joblib  # AJOUTEZ JOBLIB
import warnings
warnings.filterwarnings("ignore")

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
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Stocker l'encodeur

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
# 6️⃣ Pipeline Logistic Regression
# ==============================
pipeline_log = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_log.fit(X_train, y_train)

# Validation
y_val_pred = pipeline_log.predict(X_val)
print("\n=== Logistic Regression ===")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Test
y_test_pred = pipeline_log.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# ==============================
# 7️⃣ Pipeline Random Forest
# ==============================
pipeline_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # AJOUTER scaler pour Random Forest aussi
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline_rf.fit(X_train, y_train)

# Validation
y_val_pred_rf = pipeline_rf.predict(X_val)
print("\n=== Random Forest ===")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

# Test
y_test_pred_rf = pipeline_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print(classification_report(y_test, y_test_pred_rf))

# ==============================
# 8️⃣ Feature Importance
# ==============================
rf_model = pipeline_rf.named_steps['model']

feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ==============================
# 9️⃣ GridSearchCV (Random Forest)
# ==============================
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None],
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
# 🔟 Meilleur modèle
# ==============================
best_model = grid.best_estimator_

# Validation
y_val_best = best_model.predict(X_val)
print("\n=== Optimized Random Forest ===")
print("Validation Accuracy:", accuracy_score(y_val, y_val_best))
print(classification_report(y_val, y_val_best))

# Test
y_test_best = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_best))
print(classification_report(y_test, y_test_best))

# ==============================
# 💾 1️⃣1️⃣ Sauvegarde avec JOBLIB (RECOMMANDÉ)
# ==============================
# Sauvegarder le meilleur modèle (Random Forest optimisé)
joblib.dump(best_model, "pipeline.pkl")

# Sauvegarder aussi les encodeurs pour l'inférence
joblib.dump(encoders, "encoders.pkl")

# Sauvegarder la liste des colonnes
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\n✅ Modèle sauvegardé avec joblib!")
print("📁 Fichiers créés: pipeline.pkl, encoders.pkl, feature_columns.pkl")