"""Limpieza e ingeniería de variables (compartida por entrenamiento y app)."""

import numpy as np
import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del DataFrame con las variables derivadas.

    Va dentro del pipeline del modelo, así que la app solo tiene que
    enviar las columnas originales del cliente.
    """
    df = df.copy()

    # TotalCharges viene como texto y vacío en clientes recién llegados
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Número de servicios contratados
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    # Cargo medio mensual real y su diferencia con la cuota actual
    average = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["AvgMonthlyCharge"] = average.fillna(df["MonthlyCharges"])
    df["ChargeDiff"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

    # Perfiles de riesgo observados en el EDA
    df["FiberNoSupport"] = (
        (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
    ).astype(int)
    df["NewMonthly"] = (
        (df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)
    ).astype(int)
    df["AutoPay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["LogTenure"] = np.log1p(df["tenure"])

    return df
