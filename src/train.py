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
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)


# =====================================================
# 1. RUTAS DEL PROYECTO
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "churn_model.pkl"

RANDOM_STATE = 42

# Recall mínimo deseado al elegir el umbral (None = maximizar F1)
TARGET_RECALL = None  # por ejemplo, 0.75


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
# 4. INGENIERÍA DE VARIABLES
# =====================================================

service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

# Número de servicios contratados
df_clean["NumServices"] = (df_clean[service_cols] == "Yes").sum(axis=1)

# Cargo medio mensual real (evita dividir entre 0)
df_clean["AvgMonthlyCharge"] = df_clean["TotalCharges"] / df_clean["tenure"].replace(0, np.nan)
df_clean["AvgMonthlyCharge"] = df_clean["AvgMonthlyCharge"].fillna(df_clean["MonthlyCharges"])


# =====================================================
# 5. SEPARACIÓN DE VARIABLES
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
# 6. DIVISIÓN DE LOS DATOS
# =====================================================

# Separar el 20 % para test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Separar validación del conjunto restante
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)


# =====================================================
# 7. COMPROBACIÓN DE LAS DIVISIONES
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
# 8. IDENTIFICACIÓN DE VARIABLES
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
# 9. PREPROCESAMIENTO
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

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocesador creado correctamente")


# =====================================================
# 10. MODELOS CANDIDATOS
# =====================================================

candidates = {
    "LogReg_balanced": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "RandomForest_balanced": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


def build_pipeline(classifier):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


# =====================================================
# 11. FUNCIÓN PARA ELEGIR EL UMBRAL (SOLO CON VALIDACIÓN)
# =====================================================

def find_best_threshold(y_true, y_proba, target_recall=None):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # precisions y recalls tienen un elemento más que thresholds
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

    if target_recall is not None:
        valid = np.where(recalls >= target_recall)[0]
        if len(valid) > 0:
            # Entre los que cumplen el recall, el de mayor F1
            best_idx = valid[np.argmax(f1_scores[valid])]
            return thresholds[best_idx]

    return thresholds[np.argmax(f1_scores)]


# =====================================================
# 12. ENTRENAMIENTO Y COMPARACIÓN EN VALIDACIÓN
# =====================================================

results = {}

for name, clf in candidates.items():

    print("\n" + "=" * 50)
    print(f"MODELO: {name}")
    print("=" * 50)

    pipe = build_pipeline(clf)
    pipe.fit(X_train, y_train)

    # Probabilidades
    train_proba = pipe.predict_proba(X_train)[:, 1]
    val_proba = pipe.predict_proba(X_val)[:, 1]

    # Umbral elegido en validación
    threshold = find_best_threshold(y_val, val_proba, TARGET_RECALL)

    # Predicciones con el umbral ajustado
    train_pred = (train_proba >= threshold).astype(int)
    val_pred = (val_proba >= threshold).astype(int)

    # Métricas en validación
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred),
        "recall": recall_score(y_val, val_pred),
        "f1": f1_score(y_val, val_pred),
        "roc_auc": roc_auc_score(y_val, val_proba),
        "train_accuracy": accuracy_score(y_train, train_pred),
    }
    metrics["gap"] = abs(metrics["train_accuracy"] - metrics["accuracy"])

    results[name] = {"pipeline": pipe, "metrics": metrics, "val_pred": val_pred}

    print(f"Umbral:    {metrics['threshold']:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nINFORME DE CLASIFICACIÓN (validación)")
    print(classification_report(y_val, val_pred))

    print("MATRIZ DE CONFUSIÓN (validación)")
    print(confusion_matrix(y_val, val_pred))

    # Comprobación del overfitting
    print("\nCOMPROBACIÓN DEL OVERFITTING")
    print("----------------------------")
    print(f"Accuracy entrenamiento: {metrics['train_accuracy']:.4f}")
    print(f"Accuracy validación:     {metrics['accuracy']:.4f}")
    print(f"Diferencia:              {metrics['gap']:.4f}")

    if metrics["gap"] < 0.05:
        print("Cumple el requisito de diferencia inferior al 5 %")
    else:
        print("No cumple el requisito de diferencia inferior al 5 %")


# =====================================================
# 13. SELECCIÓN DEL MEJOR MODELO
# =====================================================

# Se prefieren los modelos que cumplen el requisito de overfitting;
# entre ellos, el de mayor F1 en validación
valid_models = {
    name: r for name, r in results.items() if r["metrics"]["gap"] < 0.05
}

pool = valid_models if valid_models else results

best_name = max(pool, key=lambda n: pool[n]["metrics"]["f1"])
best_pipeline = results[best_name]["pipeline"]
best_threshold = results[best_name]["metrics"]["threshold"]

print("\n" + "=" * 50)
print(f"MODELO SELECCIONADO: {best_name}")
print(f"Umbral seleccionado: {best_threshold:.3f}")
print("=" * 50)


# =====================================================
# 14. EVALUACIÓN FINAL EN TEST (UNA SOLA VEZ)
# =====================================================

test_proba = best_pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print("\nRESULTADOS EN TEST")
print("------------------")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, test_pred):.4f}")
print(f"F1:        {f1_score(y_test, test_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_proba):.4f}")

print("\nINFORME DE CLASIFICACIÓN (test)")
print(classification_report(y_test, test_pred))

print("MATRIZ DE CONFUSIÓN (test)")
print(confusion_matrix(y_test, test_pred))


# =====================================================
# 15. GUARDAR EL MODELO Y EL UMBRAL
# =====================================================

joblib.dump(
    {
        "model": best_pipeline,
        "threshold": best_threshold,
        "model_name": best_name
    },
    MODEL_PATH
)

print(f"\nModelo guardado en: {MODEL_PATH}")

# Para usarlo después:
# artefacto = joblib.load(MODEL_PATH)
# proba = artefacto["model"].predict_proba(X_nuevo)[:, 1]
# pred = (proba >= artefacto["threshold"]).astype(int)
