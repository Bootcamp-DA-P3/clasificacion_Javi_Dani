"""CHURN GUARD: aplicación de predicción de abandono de clientes."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "database" / "churn_guard.db"

HIGH_RISK = 0.60
NAVY, GREEN, AMBER, RED = "#0B2545", "#0F9D75", "#E8A317", "#D64545"

YES_NO = ["No", "Yes"]
LABELS = {
    "Yes": "Sí", "No": "No",
    "Female": "Mujer", "Male": "Hombre",
    "Month-to-month": "Mes a mes", "One year": "Un año", "Two year": "Dos años",
    "Fiber optic": "Fibra óptica", "DSL": "DSL",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia automática",
    "Credit card (automatic)": "Tarjeta de crédito automática",
}
INTERNET_SERVICES = {
    "OnlineSecurity": "Seguridad online",
    "OnlineBackup": "Copia de seguridad",
    "DeviceProtection": "Protección de dispositivos",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "Televisión en streaming",
    "StreamingMovies": "Películas en streaming",
}
PAYMENTS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]
ALLOWED = {
    "gender": ["Female", "Male"], "SeniorCitizen": [0, 1],
    "Partner": YES_NO, "Dependents": YES_NO, "PhoneService": YES_NO,
    "MultipleLines": YES_NO + ["No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{col: YES_NO + ["No internet service"] for col in INTERNET_SERVICES},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO, "PaymentMethod": PAYMENTS,
}
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
COLUMNS = list(ALLOWED) + NUMERIC

st.set_page_config(page_title="CHURN GUARD", page_icon="🛡️", layout="wide")

st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 2.5rem; max-width: 1200px;}}
    .hero {{background: linear-gradient(120deg, {NAVY}, #1D4ED8);
            padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;}}
    .hero h1 {{color: white; margin: 0; padding: 0; font-size: 2.1rem;}}
    .hero p {{color: #DBEAFE; margin: .3rem 0 0 0;}}
    .card {{background: white; border-radius: 14px; padding: 1.4rem;
            border: 1px solid #E2E8F0; box-shadow: 0 4px 14px rgba(15,23,42,.06);}}
    .score {{font-size: 3.2rem; font-weight: 700; line-height: 1.1;}}
    .track {{background: #E2E8F0; border-radius: 99px; height: 14px; margin: .8rem 0;}}
    .fill {{height: 14px; border-radius: 99px;}}
    </style>
    <div class="hero">
        <h1>🛡️ CHURN GUARD</h1>
        <p>Anticipa el abandono de clientes y actúa antes de perderlos</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 2. MODELO Y BASE DE DATOS
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def run_query(sql, params=()):
    DB_PATH.parent.mkdir(exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as con, con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT, datos TEXT, probabilidad REAL, umbral REAL,
                prediccion INTEGER, modelo TEXT, resultado_real INTEGER)"""
        )
        return con.execute(sql, params).fetchall()


def read_history():
    columns = ["id", "fecha", "datos", "probabilidad", "umbral",
               "prediccion", "modelo", "resultado_real"]
    rows = run_query(f"SELECT {', '.join(columns)} FROM predicciones ORDER BY id DESC")
    return pd.DataFrame(rows, columns=columns)


try:
    artifact = load_model()
except FileNotFoundError:
    st.error("No se encuentra el modelo. Ejecuta primero: python src/train.py")
    st.stop()

model = artifact["model"]
threshold = artifact["threshold"]
model_name = artifact["model_name"]


def predict(data):
    return model.predict_proba(data)[:, 1]


def risk_level(probability):
    if probability < threshold:
        return "Riesgo bajo", GREEN
    if probability < HIGH_RISK:
        return "Riesgo medio", AMBER
    return "Riesgo alto", RED


def choose(label, options, **kwargs):
    return st.selectbox(
        label, options, format_func=lambda v: LABELS.get(v, v), **kwargs
    )


tab_one, tab_batch, tab_perf, tab_feedback = st.tabs(
    ["🎯 Predicción", "📂 Predicción por lotes", "📊 Rendimiento", "💬 Feedback"]
)


# =====================================================
# 3. PREDICCIÓN INDIVIDUAL
# =====================================================

with tab_one:
    form_col, result_col = st.columns([3, 2], gap="large")
    customer = {}

    with form_col:
        with st.container(border=True):
            st.subheader("👤 Cliente")
            a, b = st.columns(2)
            with a:
                customer["gender"] = choose("Género", ["Female", "Male"])
                senior = choose("¿Es mayor de 65 años?", YES_NO)
                customer["SeniorCitizen"] = int(senior == "Yes")
            with b:
                customer["Partner"] = choose("¿Tiene pareja?", YES_NO)
                customer["Dependents"] = choose("¿Tiene personas a su cargo?", YES_NO)

        with st.container(border=True):
            st.subheader("📄 Contrato y facturación")
            a, b = st.columns(2)
            with a:
                customer["Contract"] = choose("Tipo de contrato", ALLOWED["Contract"])
                customer["PaymentMethod"] = choose("Método de pago", PAYMENTS)
                customer["PaperlessBilling"] = choose("Factura electrónica", YES_NO, index=1)
            with b:
                tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
                monthly = st.slider("Cuota mensual", 18.0, 120.0, 70.0, 0.5)
                if st.toggle("Introducir los cargos totales a mano"):
                    total = st.number_input(
                        "Cargos totales", min_value=0.0, value=float(tenure * monthly)
                    )
                else:
                    total = tenure * monthly
                    st.caption(f"Cargos totales estimados: **{total:,.2f}**")
            customer.update(tenure=tenure, MonthlyCharges=monthly, TotalCharges=total)

        with st.container(border=True):
            st.subheader("📡 Servicios")
            a, b = st.columns(2)
            with a:
                customer["PhoneService"] = choose("Teléfono", YES_NO, index=1)
                if customer["PhoneService"] == "Yes":
                    customer["MultipleLines"] = choose("Varias líneas", YES_NO)
                else:
                    customer["MultipleLines"] = "No phone service"
            with b:
                customer["InternetService"] = choose(
                    "Internet", ALLOWED["InternetService"], index=1
                )

            has_internet = customer["InternetService"] != "No"
            if not has_internet:
                st.info("Sin internet no hay servicios adicionales.")
            columns = st.columns(2)
            for position, (key, label) in enumerate(INTERNET_SERVICES.items()):
                if has_internet:
                    with columns[position % 2]:
                        active = st.toggle(label, key=key)
                    customer[key] = "Yes" if active else "No"
                else:
                    customer[key] = "No internet service"

    data = pd.DataFrame([customer])[COLUMNS]
    probability = float(predict(data)[0])
    level, color = risk_level(probability)

    with result_col:
        st.markdown(
            f"""
            <div class="card">
                <div>Probabilidad de abandono</div>
                <div class="score" style="color:{color}">{probability:.1%}</div>
                <div class="track"><div class="fill"
                     style="width:{probability:.1%}; background:{color}"></div></div>
                <h3 style="color:{color}; margin:0">{level}</h3>
                <small>Se considera en riesgo a partir del {threshold:.1%}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("🔎 Señales a vigilar")
        signals = [
            (customer["Contract"] == "Month-to-month", "Contrato mes a mes"),
            (tenure <= 12, "Cliente con un año o menos de antigüedad"),
            (customer["InternetService"] == "Fiber optic", "Tiene fibra óptica"),
            (customer["TechSupport"] == "No", "Sin soporte técnico"),
            (customer["OnlineSecurity"] == "No", "Sin seguridad online"),
            (customer["PaymentMethod"] == "Electronic check", "Paga con cheque electrónico"),
        ]
        active_signals = [text for applies, text in signals if applies]
        if active_signals:
            st.markdown("\n".join(f"- {text}" for text in active_signals))
        else:
            st.write("No presenta las señales de riesgo más habituales.")

        st.subheader("💡 Simulador de retención")
        actions = {}
        if customer["Contract"] == "Month-to-month":
            actions["Pasar a contrato de un año"] = {"Contract": "One year"}
        if customer["Contract"] != "Two year":
            actions["Pasar a contrato de dos años"] = {"Contract": "Two year"}
        if customer["TechSupport"] == "No":
            actions["Añadir soporte técnico"] = {"TechSupport": "Yes"}
        if customer["OnlineSecurity"] == "No":
            actions["Añadir seguridad online"] = {"OnlineSecurity": "Yes"}
        if "automatic" not in customer["PaymentMethod"]:
            actions["Domiciliar con tarjeta"] = {
                "PaymentMethod": "Credit card (automatic)"
            }
        discount = round(max(monthly * 0.9, 18.0), 2)
        if discount < monthly:
            actions["Aplicar un 10 % de descuento"] = {"MonthlyCharges": discount}

        if actions:
            scenarios = pd.DataFrame([{**customer, **change} for change in actions.values()])
            simulation = pd.DataFrame({
                "Acción": list(actions),
                "Nueva probabilidad": predict(scenarios[COLUMNS]) * 100,
            })
            simulation["Cambio (puntos)"] = (
                simulation["Nueva probabilidad"] - probability * 100
            )
            st.dataframe(
                simulation.sort_values("Nueva probabilidad").round(1),
                hide_index=True, width="stretch",
            )
            st.caption(
                "Cada acción se simula por separado, con la misma cuota salvo en el "
                "descuento. Es una estimación del modelo, no un efecto garantizado."
            )

        if st.button("💾 Guardar predicción", type="primary", width="stretch"):
            run_query(
                """INSERT INTO predicciones
                   (fecha, datos, probabilidad, umbral, prediccion, modelo)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(timespec="seconds"), json.dumps(customer),
                 probability, float(threshold), int(probability >= threshold), model_name),
            )
            st.success("Predicción guardada. Podrás valorarla en la pestaña Feedback.")


# =====================================================
# 4. PREDICCIÓN POR LOTES
# =====================================================

with tab_batch:
    st.subheader("Analiza una cartera de clientes")
    st.write("Sube un CSV con las mismas columnas que el conjunto de datos original.")
    st.download_button(
        "Descargar plantilla", data.to_csv(index=False),
        file_name="plantilla_clientes.csv", mime="text/csv",
    )
    uploaded = st.file_uploader("Archivo CSV", type="csv")

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [col for col in COLUMNS if col not in batch.columns]
            if missing:
                raise ValueError(f"Faltan las columnas: {', '.join(missing)}")
            if batch.empty:
                raise ValueError("El archivo no tiene clientes.")
            for col, values in ALLOWED.items():
                if not batch[col].isin(values).all():
                    raise ValueError(f"Hay valores no válidos en la columna {col}.")
            for col in NUMERIC:
                batch[col] = pd.to_numeric(batch[col], errors="coerce")
            batch["TotalCharges"] = batch["TotalCharges"].fillna(
                batch["tenure"] * batch["MonthlyCharges"]
            )
            if batch[NUMERIC].isna().any().any():
                raise ValueError("Hay valores numéricos vacíos o no válidos.")

            scores = predict(batch[COLUMNS])
            output = batch.copy()
            output.insert(0, "Probabilidad (%)", (scores * 100).round(1))
            output.insert(1, "Nivel", [risk_level(p)[0] for p in scores])
            output = output.sort_values("Probabilidad (%)", ascending=False)

            at_risk = scores >= threshold
            a, b, c = st.columns(3)
            a.metric("Clientes analizados", len(output))
            b.metric("En riesgo", int(at_risk.sum()))
            c.metric(
                "Cuota mensual en riesgo",
                f"{batch.loc[at_risk, 'MonthlyCharges'].sum():,.0f}",
            )
            st.dataframe(output, hide_index=True, width="stretch")
            st.download_button(
                "Descargar resultados", output.to_csv(index=False),
                file_name="predicciones.csv", mime="text/csv",
            )
        except ValueError as error:
            st.error(f"No se puede procesar el archivo. {error}")


# =====================================================
# 5. RENDIMIENTO DEL MODELO
# =====================================================

with tab_perf:
    st.subheader("Rendimiento en datos de test")
    st.caption(f"Modelo: {model_name} · Umbral: {threshold:.3f}")
    metrics_path = REPORTS_DIR / "metricas.json"

    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = summary["test"]
        columns = st.columns(5)
        for column, (key, label) in zip(columns, [
            ("recall", "Recall"), ("precision", "Precisión"),
            ("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC"),
        ]):
            column.metric(label, f"{test[key]:.1%}")
        st.metric(
            "Máxima diferencia train-validación (límite: 5 %)",
            f"{summary['max_diferencia_train_validacion']:.2%}",
        )
        charts = [
            "matriz_confusion.png", "curva_roc.png",
            "curva_precision_recall.png", "importancia_variables.png",
        ]
        columns = st.columns(2)
        for position, name in enumerate(charts):
            if (REPORTS_DIR / name).exists():
                columns[position % 2].image(str(REPORTS_DIR / name), width="stretch")
    else:
        st.info("Ejecuta python src/train.py para generar los informes.")


# =====================================================
# 6. FEEDBACK Y MONITORIZACIÓN
# =====================================================

with tab_feedback:
    st.subheader("¿Qué pasó realmente con el cliente?")
    history = read_history()

    if history.empty:
        st.info("Todavía no hay predicciones guardadas.")
    else:
        pending = history[history["resultado_real"].isna()]
        if not pending.empty:
            with st.form("feedback"):
                options = dict(zip(pending["id"], pending["probabilidad"]))
                selected = st.selectbox(
                    "Predicción pendiente", list(options),
                    format_func=lambda i: f"Nº {i} · probabilidad {options[i]:.1%}",
                )
                outcome = st.radio(
                    "Resultado real", ["Se quedó", "Se fue"], horizontal=True
                )
                if st.form_submit_button("Registrar resultado"):
                    run_query(
                        "UPDATE predicciones SET resultado_real = ? WHERE id = ?",
                        (int(outcome == "Se fue"), int(selected)),
                    )
                    st.rerun()

        labelled = history.dropna(subset=["resultado_real"])
        real = labelled["resultado_real"].astype(int)
        predicted = labelled["prediccion"].astype(int)
        churned = int((real == 1).sum())
        flagged = int((predicted == 1).sum())
        hits = int(((real == 1) & (predicted == 1)).sum())

        st.subheader("Rendimiento en producción")
        a, b, c, d = st.columns(4)
        a.metric("Con resultado real", f"{len(labelled)} de {len(history)}")
        b.metric("Accuracy", f"{(real == predicted).mean():.1%}" if len(labelled) else "—")
        c.metric("Recall", f"{hits / churned:.1%}" if churned else "—")
        d.metric("Precisión", f"{hits / flagged:.1%}" if flagged else "—")

        st.dataframe(history.drop(columns="datos"), hide_index=True, width="stretch")

        if len(labelled):
            new_data = pd.DataFrame([json.loads(row) for row in labelled["datos"]])
            new_data["Churn"] = real.map({0: "No", 1: "Yes"}).to_numpy()
            st.download_button(
                "Descargar datos nuevos para reentrenar", new_data.to_csv(index=False),
                file_name="datos_nuevos.csv", mime="text/csv",
            )
