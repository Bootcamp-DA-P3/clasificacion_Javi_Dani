
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. RUTAS DEL PROYECTO
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "churn_model.pkl"


# =====================================================
# 2. CARGA DEL DATASET
# =====================================================

df = pd.read_csv(DATA_PATH)

print("Dataset cargado correctamente")
print(f"Dimensiones: {df.shape}")


# =====================================================
# 3. LIMPIEZA DE DATOS
# =====================================================

df_clean = df.copy()

# Eliminar espacios sobrantes en variables de texto
for col in df_clean.select_dtypes(include=["object", "str"]).columns:
    df_clean[col] = df_clean[col].str.strip()

# Convertir cadenas vacías en valores nulos
df_clean.replace("", np.nan, inplace=True)

# Convertir TotalCharges a formato numérico
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"],
    errors="coerce"
)

# Corregir TotalCharges de clientes con tenure = 0
df_clean.loc[
    (df_clean["tenure"] == 0) &
    (df_clean["TotalCharges"].isna()),
    "TotalCharges"
] = 0

# Comprobar limpieza
print("\nValores ausentes:")
print(df_clean.isnull().sum().sum())

print("\nFilas duplicadas:")
print(df_clean.duplicated().sum())


# =====================================================
# 4. SEPARACIÓN DE VARIABLES
# =====================================================

# Variables predictoras
X = df_clean.drop(
    columns=["customerID", "Churn"]
)

# Variable objetivo
y = df_clean["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

print("\nDistribución de la variable objetivo:")
print(y.value_counts())


# =====================================================
# 5. DIVISIÓN DE LOS DATOS
# =====================================================

# Separar el 20 % para test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Separar validación del conjunto restante
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val
)


# =====================================================
# 6. COMPROBACIÓN DE LAS DIVISIONES
# =====================================================

print("\nTAMAÑO DE LOS CONJUNTOS")

print("Entrenamiento:", X_train.shape)
print("Validación:", X_val.shape)
print("Test:", X_test.shape)

print("\nDISTRIBUCIÓN DE CHURN")

print("Entrenamiento:")
print(y_train.value_counts(normalize=True))

print("\nValidación:")
print(y_val.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts(normalize=True))


# =====================================================
# 7. IDENTIFICACIÓN DE VARIABLES
# =====================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

print("\nVARIABLES NUMÉRICAS:")
print(list(numeric_features))

print("\nVARIABLES CATEGÓRICAS:")
print(list(categorical_features))


# =====================================================
# 8. TRANSFORMADORES DE PREPROCESAMIENTO
# =====================================================

# Transformador para variables numéricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Transformador para variables categóricas
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

print("\nTransformadores creados correctamente")


# =====================================================
# 9. UNIFICAR EL PREPROCESAMIENTO
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocesador creado correctamente")


# =====================================================
# 10. PIPELINE COMPLETO DEL MODELO
# =====================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

print("\nPipeline del modelo creado correctamente")


# =====================================================
# 11. ENTRENAMIENTO DEL MODELO
# =====================================================

print("\nEntrenando modelo...")

model.fit(X_train, y_train)

print("Modelo entrenado correctamente")


# =====================================================
# 12. EVALUACIÓN CON EL CONJUNTO DE VALIDACIÓN
# =====================================================

# Realizar predicciones
y_val_pred = model.predict(X_val)

# Obtener probabilidades de abandono
y_val_proba = model.predict_proba(X_val)[:, 1]

# Calcular métricas
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_proba)

print("\nRESULTADOS EN VALIDACIÓN")
print("------------------------")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nINFORME DE CLASIFICACIÓN")
print(classification_report(y_val, y_val_pred))

print("\nMATRIZ DE CONFUSIÓN")
print(confusion_matrix(y_val, y_val_pred))

# =====================================================
# 13. COMPROBACIÓN DEL OVERFITTING
# =====================================================

# Predicciones sobre entrenamiento
y_train_pred = model.predict(X_train)

# Accuracy en entrenamiento
train_accuracy = accuracy_score(y_train, y_train_pred)

# Accuracy en validación
val_accuracy = accuracy_score(y_val, y_val_pred)

# Diferencia entre ambas
overfitting_gap = abs(train_accuracy - val_accuracy)

print("\nCOMPROBACIÓN DEL OVERFITTING")
print("----------------------------")

print(f"Accuracy entrenamiento: {train_accuracy:.4f}")
print(f"Accuracy validación:     {val_accuracy:.4f}")
print(f"Diferencia:              {overfitting_gap:.4f}")

if overfitting_gap < 0.05:
    print("Cumple el requisito de diferencia inferior al 5 %")
else:
    print("No cumple el requisito de diferencia inferior al 5 %")
