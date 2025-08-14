#!/usr/bin/env python3
"""
Configurazione dual-mode per l'API (locale vs cloud).
"""

import logging
import os
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione dual-mode
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # local, cloud

if ENVIRONMENT == "cloud":
    # Configurazione per ambiente cloud (GCP)
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI", "https://mlflow-server-403815755558.europe-west1.run.app"
    )
    MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mlops-breast-cancer-models")
    DATA_BUCKET = os.getenv("DATA_BUCKET", "mlops-breast-cancer-data")
    MONITORING_BUCKET = os.getenv("MONITORING_BUCKET", "mlops-breast-cancer-monitoring")

    # Path per cloud (GCS)
    MODEL_PATH = "gs://mlops-breast-cancer-models/best_model.joblib"
    METADATA_PATH = "gs://mlops-breast-cancer-models/model_metadata.json"
    SCALER_PATH = "gs://mlops-breast-cancer-models/scaler.joblib"
    PREPROCESSING_METADATA_PATH = (
        "gs://mlops-breast-cancer-models/preprocessing_metadata.json"
    )

    logger.info("🌤️  Configurazione API: AMBIENTE CLOUD")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Model Bucket: {MODEL_BUCKET}")
    logger.info(f"Data Bucket: {DATA_BUCKET}")
    logger.info(f"Monitoring Bucket: {MONITORING_BUCKET}")
else:
    # Configurazione per ambiente locale
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_BUCKET = os.getenv("MODEL_BUCKET", "local")
    DATA_BUCKET = os.getenv("DATA_BUCKET", "local")
    MONITORING_BUCKET = os.getenv("MONITORING_BUCKET", "local")

    # Percorsi locali
    MODEL_PATH = str(
        Path(__file__).parent.parent.parent
        / "data"
        / "results"
        / "models"
        / "best_model.joblib"
    )
    METADATA_PATH = str(
        Path(__file__).parent.parent.parent
        / "data"
        / "results"
        / "models"
        / "model_metadata.json"
    )
    SCALER_PATH = str(
        Path(__file__).parent.parent.parent / "data" / "processed" / "scaler.joblib"
    )
    PREPROCESSING_METADATA_PATH = str(
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "preprocessing_metadata.json"
    )

    logger.info("🏠  Configurazione API: AMBIENTE LOCALE")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Data Bucket: {DATA_BUCKET}")
    logger.info(f"Monitoring Bucket: {MONITORING_BUCKET}")

# Configurazioni comuni
API_TITLE = "Breast Cancer Classification API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "API per la classificazione del tumore al seno utilizzando machine learning"
)


def get_api_config():
    """
    Restituisce la configurazione API per l'ambiente corrente.
    """
    config = {
        "environment": ENVIRONMENT,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_bucket": MODEL_BUCKET,
        "data_bucket": DATA_BUCKET,
        "monitoring_bucket": MONITORING_BUCKET,
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "scaler_path": str(SCALER_PATH),
        "preprocessing_metadata_path": str(PREPROCESSING_METADATA_PATH),
        "api_title": API_TITLE,
        "api_version": API_VERSION,
        "api_description": API_DESCRIPTION,
    }

    logger.info(f"Configurazione API: {config}")
    return config


def is_cloud_environment():
    """
    Verifica se siamo in ambiente cloud.
    """
    return ENVIRONMENT == "cloud"


def get_model_path():
    """
    Restituisce il path del modello per l'ambiente corrente.
    """
    return str(MODEL_PATH)


def get_mlflow_uri():
    """
    Restituisce l'URI di MLflow per l'ambiente corrente.
    """
    return MLFLOW_TRACKING_URI
