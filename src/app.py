
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# =====================================================
# 1. CONFIGURACIÓN Y RUTAS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

st.set_page_config(
    page_title="CHURN GUARD",
    page_icon="🛡️",
    layout="wide"
)


# =====================================================
# 2. CARGAR MODELO
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    artifact = load_model()

    model = artifact["model"]
    threshold = artifact["threshold"]
    model_name = artifact["model_name"]

except FileNotFoundError:
    st.error("No se encuentra el archivo del modelo.")
    st.stop()

except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# =====================================================
# 3. CABECERA
# =====================================================

st.title("🛡️ CHURN GUARD")

st.subheader("Sistema de predicción de abandono de clientes")

st.markdown(
    """
    CHURN GUARD utiliza Machine Learning para estimar
    la probabilidad de que un cliente abandone el servicio.
    """
)

st.divider()

st.caption(f"Modelo activo: {model_name}")
st.caption(f"Umbral de clasificación: {threshold:.3f}")


# =====================================================
# 4. FORMULARIO DE DATOS DEL CLIENTE
# =====================================================

st.header("Datos del cliente")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # ---------------------------------------------
    # DATOS PERSONALES
    # ---------------------------------------------

    with col1:

        st.subheader("Información personal")

        gender = st.selectbox(
            "Género",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "¿Es ciudadano senior?",
            [0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No"
        )

        partner = st.selectbox(
            "¿Tiene pareja?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "¿Tiene personas dependientes?",
            ["Yes", "No"]
        )

    # ---------------------------------------------
    # SERVICIOS CONTRATADOS
    # ---------------------------------------------

    with col2:

        st.subheader("Servicios contratados")

        phone_service = st.selectbox(
            "Servicio telefónico",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Varias líneas",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Servicio de internet",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Seguridad online",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Copia de seguridad online",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Protección de dispositivos",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Soporte técnico",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming de películas",
            ["Yes", "No", "No internet service"]
        )

    # ---------------------------------------------
    # INFORMACIÓN ECONÓMICA Y CONTRATO
    # ---------------------------------------------

    with col3:

        st.subheader("Contrato y facturación")

        contract = st.selectbox(
            "Tipo de contrato",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Facturación electrónica",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Método de pago",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        tenure = st.number_input(
            "Antigüedad (meses)",
            min_value=0,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Cargo mensual",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Cargos totales",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "Analizar riesgo de abandono",
        type="primary",
        use_container_width=True
    )


# =====================================================
# 5. PREDICCIÓN
# =====================================================

if submitted:

    # Ingeniería de variables: debe coincidir con train.py
    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    num_services = sum(
        value == "Yes" for value in service_values
    )

    avg_monthly_charge = (
        total_charges / tenure
        if tenure > 0
        else monthly_charges
    )

    # Crear dataframe con las mismas columnas del entrenamiento
    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "NumServices": num_services,
        "AvgMonthlyCharge": avg_monthly_charge
    }])

    # Obtener probabilidad de abandono
    probability = model.predict_proba(customer_data)[0][1]

    # Aplicar el umbral guardado
    prediction = int(probability >= threshold)

    st.divider()

    st.header("Resultado del análisis")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Probabilidad de abandono",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

    with col2:

        if prediction == 1:

            st.error(
                "⚠️ CLIENTE CON RIESGO DE ABANDONO"
            )

            st.write(
                "Se recomienda estudiar posibles acciones "
                "de retención para este cliente."
            )

        else:

            st.success(
                "✅ CLIENTE SIN RIESGO ELEVADO DE ABANDONO"
            )

            st.write(
                "El modelo no identifica actualmente "
                "un riesgo elevado según el umbral establecido."
            )

    st.caption(
        "La predicción es una estimación estadística, "
        "no una certeza de que el cliente vaya a abandonar."
    )
