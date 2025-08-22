#!/usr/bin/env python3
"""
FastAPI application per il deployment del modello breast cancer.
Include endpoints per predizioni, health check e monitoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import configurazione dual-mode
from src.api.config import get_api_config, is_cloud_environment

# Configurazione dual-mode
api_config = get_api_config()

# Inizializzazione FastAPI
app = FastAPI(
    title=api_config["api_title"],
    description=api_config["api_description"],
    version=api_config["api_version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelli Pydantic per request/response
class PredictionRequest(BaseModel):
    """Schema per la richiesta di predizione."""

    features: Dict[str, float] = Field(
        ...,
        description="Features del paziente per la predizione",
        examples=[
            {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave_points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
            }
        ],
    )


class PredictionResponse(BaseModel):
    """Schema per la risposta di predizione."""

    prediction: int = Field(..., description="Predizione: 0 (Benigno) o 1 (Maligno)")
    probability: float = Field(..., description="Probabilità della predizione")
    confidence: str = Field(..., description="Livello di confidenza")
    features_used: List[str] = Field(
        ..., description="Features utilizzate per la predizione"
    )
    timestamp: str = Field(..., description="Timestamp della predizione")


class HealthResponse(BaseModel):
    """Schema per health check."""

    status: str = Field(..., description="Status del servizio")
    model_loaded: bool = Field(..., description="Se il modello è caricato")
    model_version: str = Field(..., description="Versione del modello")
    timestamp: str = Field(..., description="Timestamp del check")


class BatchPredictionRequest(BaseModel):
    """Schema per predizioni batch."""

    predictions: List[Dict[str, float]] = Field(
        ...,
        description="Lista di features per predizioni multiple",
        examples=[
            {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave_points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
            }
        ],
    )


class BatchPredictionResponse(BaseModel):
    """Schema per risposta predizioni batch."""

    predictions: List[Dict[str, Any]] = Field(..., description="Lista delle predizioni")
    total_predictions: int = Field(..., description="Numero totale di predizioni")
    timestamp: str = Field(..., description="Timestamp delle predizioni")


# Variabili globali per il modello
model = None
scaler = None
feature_names = None
model_metadata = None


def load_model():
    """Carica il modello e le dipendenze."""
    global model, scaler, feature_names, model_metadata

    try:
        # Usare configurazione dual-mode
        model_path = api_config["model_path"]
        scaler_path = api_config["scaler_path"]
        metadata_path = api_config["metadata_path"]
        preprocessing_metadata_path = api_config["preprocessing_metadata_path"]

        logger.info(f"Ambiente: {api_config['environment']}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Scaler path: {scaler_path}")

        # Caricare modello
        if is_cloud_environment():
            # Per cloud, usare Google Cloud Storage
            from google.cloud import storage
            import tempfile

            # Download temporaneo del modello
            with tempfile.NamedTemporaryFile(
                suffix=".joblib", delete=False
            ) as tmp_file:
                storage_client = storage.Client()
                # Estrai bucket e blob path dal GCS path
                gcs_path = model_path.replace("gs://", "")
                bucket_name, blob_path = gcs_path.split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(tmp_file.name)
                model = joblib.load(tmp_file.name)
                logger.info(f"Modello caricato da cloud: {model_path}")
        else:
            # Per locale, usare file system
            if Path(model_path).exists():
                model = joblib.load(model_path)
                logger.info(f"Modello caricato da locale: {model_path}")
            else:
                logger.error(f"Modello non trovato: {model_path}")
                return False

        # Caricare scaler
        if is_cloud_environment():
            # Per cloud, download temporaneo
            with tempfile.NamedTemporaryFile(
                suffix=".joblib", delete=False
            ) as tmp_file:
                storage_client = storage.Client()
                # Estrai bucket e blob path dal GCS path
                gcs_path = scaler_path.replace("gs://", "")
                bucket_name, blob_path = gcs_path.split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(tmp_file.name)
                scaler = joblib.load(tmp_file.name)
                logger.info(f"Scaler caricato da cloud: {scaler_path}")
        else:
            # Per locale
            if Path(scaler_path).exists():
                scaler = joblib.load(scaler_path)
                logger.info(f"Scaler caricato da locale: {scaler_path}")
            else:
                logger.warning(f"Scaler non trovato: {scaler_path}")

        # Caricare feature names dal preprocessing metadata
        if is_cloud_environment():
            # Per cloud, download temporaneo
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
                storage_client = storage.Client()
                # Estrai bucket e blob path dal GCS path
                gcs_path = preprocessing_metadata_path.replace("gs://", "")
                bucket_name, blob_path = gcs_path.split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(tmp_file.name)
                with open(tmp_file.name, "r") as f:
                    preprocessing_metadata = json.load(f)
                    feature_names = preprocessing_metadata.get("selected_features", [])
                logger.info(
                    f"Feature names caricati da cloud: {preprocessing_metadata_path}"
                )
        else:
            # Per locale
            if Path(preprocessing_metadata_path).exists():
                with open(preprocessing_metadata_path, "r") as f:
                    preprocessing_metadata = json.load(f)
                    feature_names = preprocessing_metadata.get("selected_features", [])
                logger.info(
                    f"Feature names caricati da locale: {preprocessing_metadata_path}"
                )
            else:
                logger.warning(
                    f"Preprocessing metadata non trovato: {preprocessing_metadata_path}"
                )

        # Caricare metadata
        if is_cloud_environment():
            # Per cloud, download temporaneo
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
                storage_client = storage.Client()
                # Estrai bucket e blob path dal GCS path
                gcs_path = metadata_path.replace("gs://", "")
                bucket_name, blob_path = gcs_path.split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(tmp_file.name)
                with open(tmp_file.name, "r") as f:
                    model_metadata = json.load(f)
                logger.info(f"Metadata caricati da cloud: {metadata_path}")
        else:
            # Per locale
            if Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    model_metadata = json.load(f)
                logger.info(f"Metadata caricati da locale: {metadata_path}")
            else:
                logger.warning(f"Metadata non trovati: {metadata_path}")

        return True

    except Exception as e:
        logger.error(f"Errore nel caricamento del modello: {e}")
        return False


def preprocess_features(features: Dict[str, float]) -> np.ndarray:
    """Preprocessa le features per la predizione."""
    try:
        # Convertire in DataFrame
        df = pd.DataFrame([features])

        # Selezionare solo le features necessarie
        if feature_names:
            df = df[feature_names]

        # Scaling se disponibile
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df.values

        return df_scaled

    except Exception as e:
        logger.error(f"Errore nel preprocessing: {e}")
        raise


def get_confidence_level(probability: float) -> str:
    """Determina il livello di confidenza basato sulla probabilità."""
    if probability >= 0.9:
        return "Molto Alta"
    elif probability >= 0.8:
        return "Alta"
    elif probability >= 0.7:
        return "Media"
    elif probability >= 0.6:
        return "Bassa"
    else:
        return "Molto Bassa"


def log_prediction(features: Dict[str, float], prediction: int, probability: float):
    """Logga la predizione per audit trail."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": int(prediction),  # Converti numpy.int64 in int
        "probability": float(probability),  # Converti numpy.float64 in float
        "prediction_label": "Maligno" if prediction == 1 else "Benigno",
    }

    # Salvare log (in produzione andrebbe in database)
    log_file = Path("logs/predictions.log")
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Predizione loggata: {prediction} (prob: {probability:.4f})")


@app.on_event("startup")
async def startup_event():
    """Evento di startup dell'applicazione."""
    logger.info("🚀 Avvio Breast Cancer Classification API")

    # Caricare modello
    if not load_model():
        logger.error("❌ Impossibile caricare il modello. API non disponibile.")
    else:
        logger.info("✅ Modello caricato con successo")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Breast Cancer Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status = "healthy" if model is not None else "unhealthy"
    model_version = (
        model_metadata.get("version", "unknown") if model_metadata else "unknown"
    )

    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        model_version=model_version,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Endpoint per predizione singola."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")

    try:
        # Preprocessing features
        features_scaled = preprocess_features(request.features)

        # Predizione
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][
            1
        ]  # Probabilità classe positiva

        # Determinare confidenza
        confidence = get_confidence_level(probability)

        # Logging in background
        background_tasks.add_task(
            log_prediction, request.features, prediction, probability
        )

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            features_used=list(request.features.keys()),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Errore durante predizione: {e}")
        raise HTTPException(
            status_code=500, detail=f"Errore durante predizione: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Endpoint per predizioni batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")

    try:
        predictions = []

        for i, features in enumerate(request.predictions):
            try:
                # Preprocessing features
                features_scaled = preprocess_features(features)

                # Predizione
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0][1]

                # Determinare confidenza
                confidence = get_confidence_level(probability)

                predictions.append(
                    {
                        "id": i,
                        "prediction": int(prediction),
                        "probability": float(probability),
                        "confidence": confidence,
                        "prediction_label": "Maligno" if prediction == 1 else "Benigno",
                        "features_used": list(features.keys()),
                    }
                )

            except Exception as e:
                logger.error(f"Errore nella predizione {i}: {e}")
                predictions.append(
                    {
                        "id": i,
                        "error": str(e),
                        "prediction": None,
                        "probability": None,
                        "confidence": None,
                    }
                )

        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Errore durante predizioni batch: {e}")
        raise HTTPException(
            status_code=500, detail=f"Errore durante predizioni batch: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Informazioni sul modello."""
    if model_metadata is None:
        raise HTTPException(
            status_code=503, detail="Metadata del modello non disponibili"
        )

    return {
        "model_type": model_metadata.get("model_type", "unknown"),
        "version": model_metadata.get("version", "unknown"),
        "created_at": model_metadata.get("created_at", "unknown"),
        "hyperparameters": model_metadata.get("hyperparameters", {}),
        "validation_metrics": model_metadata.get("validation_metrics", {}),
        "features_count": len(feature_names) if feature_names else 0,
        "model_loaded": model is not None,
    }


@app.get("/features")
async def get_features():
    """Lista delle features supportate."""
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Feature names non disponibili")

    return {
        "features": feature_names,
        "count": len(feature_names),
        "description": "Features utilizzate dal modello per la classificazione",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
