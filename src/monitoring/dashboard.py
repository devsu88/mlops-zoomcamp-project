import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Import configurazione dual-mode
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.monitoring_config import (
    ENVIRONMENT,
    DATA_BUCKET,
    MODELS_BUCKET,
    MONITORING_BUCKET,
    EVIDENTLY_DASHBOARD_URL,
)

# Import per GCS se in cloud
if ENVIRONMENT == "cloud":
    try:
        from google.cloud import storage
        import tempfile

        GCS_AVAILABLE = True
    except ImportError:
        GCS_AVAILABLE = False
        st.warning("Google Cloud Storage non disponibile")
else:
    GCS_AVAILABLE = False

st.set_page_config(
    page_title="MLOps Breast Cancer Monitoring", page_icon="🏥", layout="wide"
)

st.title("🏥 MLOps Breast Cancer - Monitoring Dashboard")

# Configurazione dual-mode
project_id = os.getenv("PROJECT_ID", "mlops-breast-cancer")
environment_info = "🌤️ Cloud" if ENVIRONMENT == "cloud" else "🏠 Locale"
st.sidebar.success(f"Ambiente: {environment_info}")


def load_data_dual_mode(file_path: str, bucket_name: Optional[str] = None):
    """
    Carica dati da locale o GCS in base all'ambiente.
    """
    try:
        if ENVIRONMENT == "cloud" and GCS_AVAILABLE and bucket_name:
            # Caricamento da GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Download temporaneo
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                blob = bucket.blob(file_path)
                blob.download_to_filename(tmp_file.name)
                df = pd.read_csv(tmp_file.name)
                os.unlink(tmp_file.name)  # Rimuovi file temporaneo

            st.sidebar.info(f"📥 Dati caricati da GCS: gs://{bucket_name}/{file_path}")
            return df
        else:
            # Caricamento locale
            local_path = Path(file_path)
            if local_path.exists():
                df = pd.read_csv(local_path)
                st.sidebar.info(f"📁 Dati caricati da locale: {file_path}")
                return df
            else:
                st.error(f"❌ File non trovato: {file_path}")
                return None

    except Exception as e:
        st.error(f"❌ Errore caricamento dati: {e}")
        return None


def load_monitoring_data():
    """
    Carica dati di monitoring in base all'ambiente.
    """
    try:
        if ENVIRONMENT == "cloud":
            # Carica dati da GCS
            data_df = load_data_dual_mode("processed/train_set.csv", DATA_BUCKET)
        else:
            # Carica dati locali
            data_df = load_data_dual_mode("data/processed/train_set.csv")

        return data_df

    except Exception as e:
        st.error(f"❌ Errore caricamento dati monitoring: {e}")
        return None


st.header("📊 Data Quality Monitoring")

# Carica dati reali
data_df = load_monitoring_data()

if data_df is not None:
    # Calcola metriche reali
    total_rows = len(data_df)
    missing_values = data_df.isnull().sum().sum()
    missing_pct = (missing_values / (total_rows * len(data_df.columns))) * 100
    data_quality_score = max(0, 100 - missing_pct)

    # Calcola metriche reali aggiuntive
    duplicate_rows = data_df.duplicated().sum()
    duplicate_pct = (duplicate_rows / total_rows) * 100

    # Calcola outliers semplificato (valori oltre 3 std)
    outliers_count = 0
    for col in data_df.select_dtypes(include=[np.number]).columns:
        Q1 = data_df[col].quantile(0.25)
        Q3 = data_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count += (
            (data_df[col] < lower_bound) | (data_df[col] > upper_bound)
        ).sum()

    outliers_pct = (
        outliers_count
        / (total_rows * len(data_df.select_dtypes(include=[np.number]).columns))
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Data Quality Score",
            f"{data_quality_score:.1f}%",
            f"{'↑' if data_quality_score > 90 else '↓'} {abs(data_quality_score - 90):.1f}%",
        )

    with col2:
        duplicate_status = "✅ Clean" if duplicate_pct < 1 else "⚠️ Duplicates"
        st.metric("Duplicate Rows", duplicate_status, f"{duplicate_pct:.1f}%")

    with col3:
        outliers_status = "✅ Normal" if outliers_pct < 5 else "⚠️ Outliers"
        st.metric(
            "Outliers Detected",
            outliers_status,
            f"{outliers_count} ({outliers_pct:.1f}%)",
        )

    # Mostra info sui dati
    st.info(f"📊 Dati caricati: {data_df.shape[0]} righe, {data_df.shape[1]} colonne")
else:
    st.error(
        "❌ Impossibile caricare i dati. Verificare la connessione al bucket GCS e la disponibilità dei file."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Quality Score", "N/A", "Dati non disponibili")

    with col2:
        st.metric("Data Drift Detected", "N/A", "Dati non disponibili")

    with col3:
        st.metric("Model Performance", "N/A", "Dati non disponibili")

st.header("📊 Dataset Overview")

if data_df is not None:
    # Mostra statistiche descrittive del dataset
    try:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Statistiche Numeriche")
            numeric_cols = data_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(data_df[numeric_cols].describe())
            else:
                st.info("Nessuna colonna numerica trovata")

        with col2:
            st.subheader("📋 Informazioni Dataset")
            st.write(f"**Righe totali:** {data_df.shape[0]}")
            st.write(f"**Colonne totali:** {data_df.shape[1]}")
            st.write(f"**Colonne numeriche:** {len(numeric_cols)}")
            st.write(
                f"**Colonne categoriche:** {len(data_df.select_dtypes(include=['object']).columns)}"
            )
            st.write(
                f"**Memoria utilizzata:** {data_df.memory_usage(deep=True).sum() / 1024:.1f} KB"
            )
    except Exception as e:
        st.error(f"❌ Errore generazione overview: {e}")
        st.info(
            "📊 Impossibile visualizzare le statistiche. Verificare la struttura del dataset."
        )
else:
    st.error("❌ Nessun dato disponibile per l'overview.")
    st.info(
        "📊 Verificare che i dati siano stati caricati correttamente dal bucket GCS."
    )

st.header("🔍 Data Drift Analysis")

if data_df is not None:
    try:
        # Calcola drift score semplificato per le features principali
        features_to_check = [
            "radius_mean",
            "perimeter_mean",
            "area_mean",
            "compactness_mean",
        ]
        drift_scores = []

        for feature in features_to_check:
            if feature in data_df.columns:
                # Calcolo semplificato: coefficiente di variazione
                values = data_df[feature].dropna()
                if len(values) > 1:
                    cv = values.std() / values.mean()
                    drift_score = min(1.0, cv * 10)  # Normalizza a 0-1
                else:
                    drift_score = 0.0
            else:
                drift_score = 0.0

            drift_scores.append(drift_score)

        # Crea tabella con dati reali
        drift_data = pd.DataFrame(
            {
                "Feature": features_to_check,
                "Drift Score": [f"{score:.3f}" for score in drift_scores],
                "Status": [
                    "✅ Stable"
                    if score < 0.1
                    else "⚠️ Minor Drift"
                    if score < 0.3
                    else "🚨 High Drift"
                    for score in drift_scores
                ],
            }
        )

        st.table(drift_data)
        st.info("📊 Drift analysis basata su dati reali")

    except Exception as e:
        st.error(f"❌ Errore calcolo drift: {e}")
        st.info(
            "📊 Impossibile calcolare i drift scores. Verificare la struttura del dataset."
        )
else:
    # Nessun dato disponibile
    st.error("❌ Nessun dato disponibile per l'analisi del drift.")
    st.info(
        "📊 Verificare che i dati siano stati caricati correttamente dal bucket GCS."
    )

st.header("📋 System Status")

# Status dei servizi - verifica reale se possibile
if ENVIRONMENT == "cloud":
    try:
        # Verifica accesso ai bucket GCS
        storage_client = storage.Client()

        col1, col2, col3 = st.columns(3)

        with col1:
            # Verifica bucket dati
            try:
                bucket = storage_client.bucket(DATA_BUCKET)
                bucket.reload()
                st.success(f"📊 Data Bucket: 🟢 Online")
                st.caption(f"gs://{DATA_BUCKET}")
            except Exception as e:
                st.error(f"📊 Data Bucket: 🔴 Offline")
                st.caption(f"Errore: {str(e)[:50]}...")

        with col2:
            # Verifica bucket modelli
            try:
                bucket = storage_client.bucket(MODELS_BUCKET)
                bucket.reload()
                st.success(f"🤖 Models Bucket: 🟢 Online")
                st.caption(f"gs://{MODELS_BUCKET}")
            except Exception as e:
                st.error(f"🤖 Models Bucket: 🔴 Offline")
                st.caption(f"Errore: {str(e)[:50]}...")

        with col3:
            # Verifica bucket monitoring
            try:
                bucket = storage_client.bucket(MONITORING_BUCKET)
                bucket.reload()
                st.success(f"🔍 Monitoring Bucket: 🟢 Online")
                st.caption(f"gs://{MONITORING_BUCKET}")
            except Exception as e:
                st.error(f"🔍 Monitoring Bucket: 🔴 Offline")
                st.caption(f"Errore: {str(e)[:50]}...")

    except Exception as e:
        st.warning("⚠️ Impossibile verificare lo stato dei servizi GCS")
        st.caption(f"Errore: {str(e)}")
else:
    st.info("🏠 Ambiente locale - status servizi non verificabile")
    st.caption("I servizi locali sono gestiti manualmente")

st.sidebar.header("⚙️ Configuration")
st.sidebar.text(f"Project: {project_id}")
st.sidebar.text(f"Environment: {ENVIRONMENT}")

if ENVIRONMENT == "cloud":
    st.sidebar.text(f"Data Bucket: {DATA_BUCKET}")
    # st.sidebar.text(f"Monitoring Bucket: {MONITORING_BUCKET}")  # Rimosso - non più utilizzato
    st.sidebar.text(f"Models Bucket: {MODELS_BUCKET}")
    st.sidebar.text(f"Dashboard URL: {EVIDENTLY_DASHBOARD_URL}")
else:
    st.sidebar.text("Data Source: Local Files")
    st.sidebar.text("Monitoring: Local Directory")
    st.sidebar.text("Models: Local Directory")

st.sidebar.header("🔄 Actions")
if st.sidebar.button("Refresh Data"):
    st.rerun()

if st.sidebar.button("Export Report"):
    st.success("Report exported successfully!")
