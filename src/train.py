"""Entrenamiento de CHURN GUARD.

Uso:  python src/train.py
Genera el modelo en models/ y las métricas y gráficas en reports/.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, brier_score_loss, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import add_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TARGET_RECALL = 0.75   # detectar al menos el 75 % de los abandonos
MAX_GAP = 0.05         # sobreajuste máximo permitido


def build_pipeline(classifier):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def search_model(classifier, params, X, y, cv):
    """Busca hiperparámetros y elige el mejor AUC sin sobreajuste."""
    search = RandomizedSearchCV(
        build_pipeline(classifier),
        {f"classifier__{k}": v for k, v in params.items()},
        n_iter=20, scoring="roc_auc", cv=cv, return_train_score=True,
        random_state=RANDOM_STATE, n_jobs=-1, refit=False,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results["gap"] = results["mean_train_score"] - results["mean_test_score"]
    # Margen de seguridad: exigimos menos de 3 puntos de diferencia en AUC
    valid = results[results["gap"] < 0.03]
    if valid.empty:
        valid = results.nsmallest(1, "gap")
    best = valid.loc[valid["mean_test_score"].idxmax()]

    pipeline = build_pipeline(classifier).set_params(**best["params"])
    summary = {
        "cv_auc": best["mean_test_score"],
        "cv_auc_std": best["std_test_score"],
        "train_auc": best["mean_train_score"],
        "gap_auc": best["gap"],
        "params": best["params"],
    }
    return pipeline, summary


def choose_threshold(y_true, proba, target_recall):
    """Umbral más alto que alcanza el recall objetivo."""
    _, recalls, thresholds = precision_recall_curve(y_true, proba)
    return float(thresholds[recalls[:-1] >= target_recall].max())


def get_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def cross_validated_check(pipeline, X, y, cv, threshold):
    """Compara las métricas de train y validación en cada partición."""
    train_scores, val_scores = [], []
    for train_idx, val_idx in cv.split(X, y):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        for idx, scores in ((train_idx, train_scores), (val_idx, val_scores)):
            proba = pipeline.predict_proba(X.iloc[idx])[:, 1]
            scores.append(get_metrics(y.iloc[idx], proba, threshold))
    table = pd.DataFrame({
        "train": pd.DataFrame(train_scores).mean(),
        "validacion": pd.DataFrame(val_scores).mean(),
    })
    table["diferencia"] = table["train"] - table["validacion"]
    return table


def save_figure(name):
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / name, dpi=150)
    plt.close()


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Datos: el test se aparta y solo se usa al final
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. Modelos candidatos y sus hiperparámetros
    candidates = {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000),
            {"C": list(np.logspace(-3, 2, 40))},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [200, 300], "max_depth": [4, 5, 6, 7],
             "min_samples_leaf": [10, 20, 40], "max_features": ["sqrt", 0.5]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 150, 200, 300], "max_depth": [1, 2, 3],
             "learning_rate": [0.01, 0.02, 0.03, 0.05],
             "min_samples_leaf": [10, 20, 40], "subsample": [0.7, 0.8, 1.0]},
        ),
    }

    pipelines, comparison = {}, {}
    for name, (classifier, params) in candidates.items():
        print(f"Optimizando {name}...")
        pipelines[name], comparison[name] = search_model(
            classifier, params, X_train, y_train, cv
        )

    table = pd.DataFrame(comparison).T
    table.to_csv(REPORTS_DIR / "comparacion_modelos.csv")
    print("\nCOMPARACIÓN (validación cruzada)")
    print(table.drop(columns="params").astype(float).round(4))

    # Entre los modelos de conjunto y la regresión, gana el mayor AUC
    best_name = max(comparison, key=lambda n: comparison[n]["cv_auc"])
    model = pipelines[best_name]
    print(f"\nModelo seleccionado: {best_name}")
    print(comparison[best_name]["params"])

    # 3. Umbral calculado con predicciones fuera de muestra (sin tocar test)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof, TARGET_RECALL)
    print(f"Umbral: {threshold:.3f}")

    # 4. Sobreajuste con el umbral final
    overfit = cross_validated_check(model, X_train, y_train, cv, threshold)
    overfit.to_csv(REPORTS_DIR / "sobreajuste.csv")
    print("\nSOBREAJUSTE")
    print(overfit.round(4))
    max_gap = overfit.drop(index="brier")["diferencia"].abs().max()
    print(f"Máxima diferencia: {max_gap:.2%} "
          f"({'CUMPLE' if max_gap < MAX_GAP else 'NO CUMPLE'})")

    # 5. Evaluación final en test
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test_metrics = get_metrics(y_test, proba, threshold)
    print("\nTEST")
    for key, value in test_metrics.items():
        print(f"{key:10s} {value:.4f}")

    # 6. Gráficas para el informe
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Se queda", "Se va"], cmap="Blues"
    )
    plt.title("Matriz de confusión (test)")
    save_figure("matriz_confusion.png")

    RocCurveDisplay.from_predictions(y_test, proba, name=best_name)
    plt.title("Curva ROC (test)")
    save_figure("curva_roc.png")

    PrecisionRecallDisplay.from_predictions(y_test, proba, name=best_name)
    plt.title("Curva precisión-recall (test)")
    save_figure("curva_precision_recall.png")

    # Importancia de las variables originales (por permutación)
    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=20, random_state=RANDOM_STATE,
    )
    ranking = pd.Series(importance.importances_mean, index=X_test.columns)
    ranking.sort_values(ascending=False).to_csv(
        REPORTS_DIR / "importancia_variables.csv", header=["caida_auc"]
    )
    ranking.sort_values().tail(10).plot.barh(figsize=(8, 5))
    plt.title("Variables más importantes")
    plt.xlabel("Caída del ROC-AUC al desordenar la variable")
    save_figure("importancia_variables.png")

    # 7. Análisis de errores
    errors = X_test.copy()
    errors["real"] = y_test
    errors["probabilidad"] = proba
    errors["resultado"] = np.select(
        [(y_test == 1) & (pred == 1), (y_test == 0) & (pred == 0),
         (y_test == 0) & (pred == 1)],
        ["acierto_se_va", "acierto_se_queda", "falsa_alarma"],
        default="fuga_no_detectada",
    )
    errors.to_csv(REPORTS_DIR / "predicciones_test.csv", index=False)
    profile = errors.groupby("resultado")[["tenure", "MonthlyCharges", "probabilidad"]].mean()
    profile["clientes"] = errors["resultado"].value_counts()
    profile.to_csv(REPORTS_DIR / "analisis_errores.csv")
    print("\nANÁLISIS DE ERRORES")
    print(profile.round(2))

    # 8. Guardar modelo y resumen
    joblib.dump(
        {"model": model, "threshold": threshold, "model_name": best_name},
        MODEL_PATH, compress=3,
    )
    summary = {
        "modelo": best_name,
        "umbral": threshold,
        "recall_objetivo": TARGET_RECALL,
        "test": test_metrics,
        "max_diferencia_train_validacion": max_gap,
    }
    (REPORTS_DIR / "metricas.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
