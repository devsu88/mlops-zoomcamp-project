"""
Batch monitoring function for Evidently AI
This function is triggered to run monitoring checks on data
Supports dual-mode: local vs cloud
"""

import json
import logging
import os
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configurazione dual-mode
from monitoring.monitoring_config import (
    ENVIRONMENT,
    DATA_BUCKET,
    MONITORING_BUCKET,
    MODELS_BUCKET,
    DATA_QUALITY_THRESHOLDS,
    DRIFT_THRESHOLDS,
    PERFORMANCE_THRESHOLDS,
)

# Import per GCS se in cloud
if ENVIRONMENT == "cloud":
    try:
        from google.cloud import storage
        import tempfile

        GCS_AVAILABLE = True
    except ImportError:
        GCS_AVAILABLE = False
        logger.warning("Google Cloud Storage non disponibile")
else:
    GCS_AVAILABLE = False


def load_data_dual_mode(file_path: str, bucket_name: str | None = None):
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

            logger.info(f"📥 Dati caricati da GCS: gs://{bucket_name}/{file_path}")
            return df
        else:
            # Caricamento locale
            local_path = Path(file_path)
            if local_path.exists():
                df = pd.read_csv(local_path)
                logger.info(f"📁 Dati caricati da locale: {file_path}")
                return df
            else:
                logger.warning(f"⚠️ File non trovato: {file_path}")
                return None

    except Exception as e:
        logger.error(f"❌ Errore caricamento dati: {e}")
        return None


def perform_data_quality_check():
    """
    Esegue controllo qualità dati.
    """
    try:
        logger.info("🔍 Esecuzione data quality check...")

        # Carica dati di riferimento
        if ENVIRONMENT == "cloud":
            reference_data = load_data_dual_mode("processed/train_set.csv", DATA_BUCKET)
        else:
            reference_data = load_data_dual_mode("data/processed/train_set.csv")

        if reference_data is None:
            return {
                "status": "error",
                "message": "Impossibile caricare dati di riferimento",
            }

        # Calcola metriche qualità
        total_rows = len(reference_data)
        missing_values = reference_data.isnull().sum().sum()
        missing_pct = (
            missing_values / (total_rows * len(reference_data.columns))
        ) * 100

        # Valida soglie
        quality_score = max(0, 100 - missing_pct)
        quality_status = (
            "good"
            if quality_score >= 90
            else "warning"
            if quality_score >= 70
            else "critical"
        )

        return {
            "status": "success",
            "quality_score": quality_score,
            "missing_percentage": missing_pct,
            "total_rows": total_rows,
            "quality_status": quality_status,
            "threshold": DATA_QUALITY_THRESHOLDS["missing_values_threshold"] * 100,
        }

    except Exception as e:
        logger.error(f"❌ Errore data quality check: {e}")
        return {"status": "error", "message": str(e)}


def perform_drift_detection():
    """
    Esegue drift detection.
    """
    try:
        logger.info("🔄 Esecuzione drift detection...")

        # Carica dati di riferimento e attuali
        if ENVIRONMENT == "cloud":
            reference_data = load_data_dual_mode("processed/train_set.csv", DATA_BUCKET)
            current_data = load_data_dual_mode("processed/test_set.csv", DATA_BUCKET)
        else:
            reference_data = load_data_dual_mode("data/processed/train_set.csv")
            current_data = load_data_dual_mode("data/processed/test_set.csv")

        if reference_data is None or current_data is None:
            return {
                "status": "error",
                "message": "Impossibile caricare dati per drift detection",
            }

        # Calcolo drift semplificato (coefficiente di variazione)
        drift_scores = {}
        features_to_check = ["radius_mean", "perimeter_mean", "area_mean"]

        for feature in features_to_check:
            if feature in reference_data.columns and feature in current_data.columns:
                ref_values = reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()

                if len(ref_values) > 1 and len(curr_values) > 1:
                    ref_cv = ref_values.std() / ref_values.mean()
                    curr_cv = curr_values.std() / curr_values.mean()
                    drift_score = abs(curr_cv - ref_cv) / max(ref_cv, 0.001)
                    drift_scores[feature] = min(1.0, drift_score)
                else:
                    drift_scores[feature] = 0.0
            else:
                drift_scores[feature] = 0.0

        # Valuta drift complessivo
        avg_drift = sum(drift_scores.values()) / len(drift_scores)
        drift_status = (
            "stable" if avg_drift < 0.1 else "minor" if avg_drift < 0.3 else "high"
        )

        return {
            "status": "success",
            "drift_scores": drift_scores,
            "average_drift": avg_drift,
            "drift_status": drift_status,
            "threshold": DRIFT_THRESHOLDS["psi_threshold"],
        }

    except Exception as e:
        logger.error(f"❌ Errore drift detection: {e}")
        return {"status": "error", "message": str(e)}


def save_monitoring_results(results: Dict[str, Any]):
    """
    Salva risultati monitoring su locale o GCS.
    """
    try:
        timestamp = datetime.now().isoformat()
        results["timestamp"] = timestamp
        results["environment"] = ENVIRONMENT

        # Salva localmente
        results_path = (
            Path(__file__).parent.parent.parent / "monitoring" / "batch_results.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Risultati salvati localmente: {results_path}")

        # Salva su GCS se in cloud
        if ENVIRONMENT == "cloud" and GCS_AVAILABLE:
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(MONITORING_BUCKET)

                # Upload risultati
                blob = bucket.blob("batch_results.json")
                blob.upload_from_string(
                    json.dumps(results, indent=2, default=str),
                    content_type="application/json",
                )

                logger.info(
                    f"🌤️ Risultati caricati su GCS: gs://{MONITORING_BUCKET}/batch_results.json"
                )

            except Exception as e:
                logger.error(f"❌ Errore upload GCS: {e}")
                logger.warning("Continuando con salvataggio locale...")

        return True

    except Exception as e:
        logger.error(f"❌ Errore salvataggio risultati: {e}")
        return False


def run_monitoring(request):
    """
    Main function for batch monitoring
    Triggered by HTTP request or Cloud Scheduler
    """
    try:
        logger.info(f"🚀 Avvio batch monitoring - Ambiente: {ENVIRONMENT}")

        # Parse request data
        request_json = request.get_json(silent=True)

        if request_json:
            logger.info(f"📥 Richiesta ricevuta: {request_json}")
        else:
            logger.info("📥 Nessun dato JSON nella richiesta")

        # Esegui controlli monitoring
        monitoring_results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT,
            "checks_performed": [],
            "results": {},
        }

        # Data Quality Check
        quality_result = perform_data_quality_check()
        monitoring_results["results"]["data_quality"] = quality_result
        monitoring_results["checks_performed"].append("data_quality_check")

        # Drift Detection
        drift_result = perform_drift_detection()
        monitoring_results["results"]["drift_detection"] = drift_result
        monitoring_results["checks_performed"].append("drift_detection")

        # Salva risultati
        save_success = save_monitoring_results(monitoring_results)
        if not save_success:
            logger.warning("⚠️ Impossibile salvare risultati monitoring")

        logger.info(
            f"✅ Monitoring completato: {len(monitoring_results['checks_performed'])} controlli eseguiti"
        )

        # Return success response
        return json.dumps(monitoring_results), 200, {"Content-Type": "application/json"}

    except Exception as e:
        logger.error(f"❌ Errore nella funzione monitoring: {str(e)}")
        error_response = {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT,
        }
        return json.dumps(error_response), 500, {"Content-Type": "application/json"}


# For local testing
if __name__ == "__main__":

    class MockRequest:
        def get_json(self, silent=True):
            return {"test": "data"}

    result = run_monitoring(MockRequest())
    print(result)
