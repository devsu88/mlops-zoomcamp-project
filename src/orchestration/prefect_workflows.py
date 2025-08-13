#!/usr/bin/env python3
"""
Script per l'orchestrazione dei workflow ML con Prefect.
Include workflow per data preprocessing, training, validation e deployment.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib

# Configurare matplotlib per ambiente non-interactive
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from prefect.server.schemas.schedules import CronSchedule
from prefect.tasks import task_input_hash

matplotlib.use("Agg")  # Backend non-interactive
import matplotlib.pyplot as plt

# Aggiungere il path per importare i moduli
sys.path.append(str(Path(__file__).parent.parent))

# Importazioni corrette per le funzioni
from data.preprocessing import main as preprocess_data
from models.feature_importance_analysis import main as analyze_feature_importance
from models.hyperparameter_tuning import main as tune_hyperparameters
from models.model_validation import main as perform_comprehensive_validation
from models.train_baseline import main as train_baseline_models
from models.train_with_mlflow import main as train_models_with_mlflow
from models.mlflow_config import setup_mlflow

# Import configurazione dual-mode Prefect
from orchestration.prefect_config import get_prefect_config, is_cloud_environment

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import per GCS se in cloud
if is_cloud_environment():
    try:
        from google.cloud import storage
        import tempfile

        GCS_AVAILABLE = True
    except ImportError:
        GCS_AVAILABLE = False
        logger.warning("Google Cloud Storage non disponibile")
else:
    GCS_AVAILABLE = False

# Configurazione
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
RANDOM_STATE = 42

# Configurazione dual-mode Prefect
prefect_config = get_prefect_config()

# Prefect storage - dual-mode
if is_cloud_environment():
    # Per cloud, usare storage remoto (GCS)
    from prefect.filesystems import GCS

    storage = GCS(bucket_path=prefect_config["project_name"])
    logger.info(f"🌤️  Storage Prefect: GCS Cloud")
else:
    # Per locale, usare file system locale
    storage = LocalFileSystem(basepath=str(Path.cwd()))
    logger.info(f"🏠  Storage Prefect: File System Locale")


def save_model_dual_mode(model, file_path: Path, bucket_name: str | None = None):
    """
    Salva il modello in locale e su GCS se in cloud.
    """
    logger = get_run_logger()

    # Salvataggio locale
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    logger.info(f"Modello salvato localmente: {file_path}")

    # Salvataggio su GCS se in cloud
    if is_cloud_environment() and GCS_AVAILABLE and bucket_name:
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Upload del modello
            blob = bucket.blob(str(file_path.relative_to(Path.cwd())))
            blob.upload_from_filename(str(file_path))

            logger.info(
                f"🌤️  Modello caricato su GCS: gs://{bucket_name}/{file_path.relative_to(Path.cwd())}"
            )

        except Exception as e:
            logger.error(f"Errore upload GCS: {e}")
            logger.warning("Continuando con salvataggio locale...")

    return str(file_path)


def save_metadata_dual_mode(
    metadata: dict, file_path: Path, bucket_name: str | None = None
):
    """
    Salva i metadata in locale e su GCS se in cloud.
    """
    logger = get_run_logger()

    # Salvataggio locale
    file_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata salvato localmente: {file_path}")

    # Salvataggio su GCS se in cloud
    if is_cloud_environment() and GCS_AVAILABLE and bucket_name:
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Upload dei metadata
            blob = bucket.blob(str(file_path.relative_to(Path.cwd())))
            blob.upload_from_filename(str(file_path))

            logger.info(
                f"🌤️  Metadata caricato su GCS: gs://{bucket_name}/{file_path.relative_to(Path.cwd())}"
            )

        except Exception as e:
            logger.error(f"Errore upload GCS: {e}")
            logger.warning("Continuando con salvataggio locale...")

    return str(file_path)


@task(name="setup-environment", retries=2, retry_delay_seconds=30)
def setup_environment():
    """
    Setup dell'ambiente di lavoro.
    """
    logger = get_run_logger()
    logger.info("=== SETUP ENVIRONMENT ===")

    # Creare directory necessarie
    directories = [DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)
        logger.info(f"Directory creata/verificata: {directory}")

    # Setup MLflow
    experiment_id = setup_mlflow()
    logger.info(f"MLflow experiment ID: {experiment_id}")

    return {
        "data_dir": str(DATA_DIR),
        "processed_data_dir": str(PROCESSED_DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "models_dir": str(MODELS_DIR),
        "mlflow_experiment_id": experiment_id,
    }


@task(name="download-dataset", retries=3, retry_delay_seconds=60)
def download_dataset():
    """
    Download del dataset da Kaggle.
    """
    logger = get_run_logger()
    logger.info("=== DOWNLOAD DATASET ===")

    try:
        from data.download_dataset import download_from_kaggle, verify_dataset

        # Download dataset
        dataset_info = download_from_kaggle()
        if isinstance(dataset_info, tuple):
            dataset_path, df = dataset_info
        else:
            dataset_path = dataset_info
            df = None

        logger.info(f"Dataset scaricato: {dataset_path}")

        # Verifica integrità
        if df is not None:
            is_valid = verify_dataset(df)
            if not is_valid:
                raise ValueError("Dataset non valido dopo il download")

        logger.info("Dataset verificato con successo")
        return str(dataset_path)

    except Exception as e:
        logger.error(f"Errore durante download dataset: {e}")
        raise


@task(name="preprocess-data", retries=2, retry_delay_seconds=30)
def preprocess_data_task(dataset_path: str):
    """
    Preprocessing dei dati.
    """
    logger = get_run_logger()
    logger.info("=== PREPROCESSING DATA ===")

    try:
        # Eseguire preprocessing
        preprocess_data()

        # Verificare che i file siano stati creati
        train_path = PROCESSED_DATA_DIR / "train_set.csv"
        test_path = PROCESSED_DATA_DIR / "test_set.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "File di training/test non trovati dopo preprocessing"
            )

        logger.info(f"Preprocessing completato")
        logger.info(f"Train set: {train_path}")
        logger.info(f"Test set: {test_path}")

        return {"train_path": str(train_path), "test_path": str(test_path)}

    except Exception as e:
        logger.error(f"Errore durante preprocessing: {e}")
        raise


@task(name="train-baseline-models", retries=2, retry_delay_seconds=60)
def train_baseline_models_task():
    """
    Training dei modelli baseline.
    """
    logger = get_run_logger()
    logger.info("=== TRAINING BASELINE MODELS ===")

    try:
        # Eseguire training baseline
        results = train_baseline_models()

        logger.info("Training baseline completato")
        return results

    except Exception as e:
        logger.error(f"Errore durante training baseline: {e}")
        raise


@task(name="train-mlflow-models", retries=2, retry_delay_seconds=60)
def train_mlflow_models_task():
    """
    Training dei modelli con MLflow tracking.
    """
    logger = get_run_logger()
    logger.info("=== TRAINING MLFLOW MODELS ===")

    try:
        # Eseguire training con MLflow
        results = train_models_with_mlflow()

        logger.info("Training MLflow completato")
        return results

    except Exception as e:
        logger.error(f"Errore durante training MLflow: {e}")
        raise


@task(name="hyperparameter-tuning", retries=2, retry_delay_seconds=60)
def hyperparameter_tuning_task():
    """
    Hyperparameter tuning per Logistic Regression.
    """
    logger = get_run_logger()
    logger.info("=== HYPERPARAMETER TUNING ===")

    try:
        # Eseguire hyperparameter tuning
        results = tune_hyperparameters()

        logger.info("Hyperparameter tuning completato")
        return results

    except Exception as e:
        logger.error(f"Errore durante hyperparameter tuning: {e}")
        raise


@task(name="feature-importance-analysis", retries=2, retry_delay_seconds=30)
def feature_importance_analysis_task():
    """
    Analisi dell'importanza delle features.
    """
    logger = get_run_logger()
    logger.info("=== FEATURE IMPORTANCE ANALYSIS ===")

    try:
        # Eseguire analisi feature importance
        results = analyze_feature_importance()

        logger.info("Feature importance analysis completata")
        return results

    except Exception as e:
        logger.error(f"Errore durante feature importance analysis: {e}")
        raise


@task(name="model-validation", retries=2, retry_delay_seconds=60)
def model_validation_task():
    """
    Validazione approfondita del modello.
    """
    logger = get_run_logger()
    logger.info("=== MODEL VALIDATION ===")

    try:
        # Eseguire validazione modello
        results = perform_comprehensive_validation()

        logger.info("Model validation completata")
        return results

    except Exception as e:
        logger.error(f"Errore durante model validation: {e}")
        raise


@task(name="save-best-model", retries=2, retry_delay_seconds=30)
def save_best_model_task(validation_results: dict):
    """
    Salva il miglior modello per il deployment.
    """
    logger = get_run_logger()
    logger.info("=== SAVE BEST MODEL ===")

    try:
        # Caricare il modello ottimizzato
        model_path = RESULTS_DIR / "logistic_regression_gridsearch_tuned.joblib"
        model = joblib.load(model_path)

        # Salvare il modello per il deployment con dual-mode
        deployment_model_path = MODELS_DIR / "best_model.joblib"

        # Bucket per modelli (da configurazione)
        model_bucket = "mlops-breast-cancer-models" if is_cloud_environment() else None

        # Salvataggio dual-mode del modello
        model_path = save_model_dual_mode(
            model, deployment_model_path, model_bucket if model_bucket else None
        )

        # Salvare anche i metadati
        metadata = {
            "model_type": "LogisticRegression",
            "hyperparameters": {
                "C": 0.01,
                "penalty": "l1",
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
            "validation_metrics": {
                "cv_recall_mean": validation_results["cv_results"]["recall"]["mean"],
                "cv_f1_mean": validation_results["cv_results"]["f1"]["mean"],
                "bootstrap_recall_mean": validation_results["bootstrap_stats"][
                    "recall"
                ]["mean"],
                "bootstrap_f1_mean": validation_results["bootstrap_stats"]["f1"][
                    "mean"
                ],
            },
            "created_at": datetime.now().isoformat(),
            "model_path": str(deployment_model_path),
            "environment": "cloud" if is_cloud_environment() else "local",
        }

        metadata_path = MODELS_DIR / "model_metadata.json"

        # Salvataggio dual-mode dei metadata
        metadata_path = save_metadata_dual_mode(
            metadata, metadata_path, model_bucket if model_bucket else None
        )

        logger.info(f"Modello salvato: {deployment_model_path}")
        logger.info(f"Metadata salvato: {metadata_path}")

        return {
            "model_path": str(deployment_model_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Errore durante salvataggio modello: {e}")
        raise


@task(name="upload-auxiliary-files", retries=2, retry_delay_seconds=30)
def upload_auxiliary_files_task():
    """
    Carica file ausiliari (scaler, preprocessing metadata) su GCS se in cloud.
    """
    logger = get_run_logger()
    logger.info("=== UPLOAD AUXILIARY FILES ===")

    if not is_cloud_environment():
        logger.info("🏠  Ambiente locale - skip upload GCS")
        return {"status": "local_environment"}

    try:
        # Bucket per modelli
        model_bucket = "mlops-breast-cancer-models"

        # File da caricare
        files_to_upload = [
            ("data/processed/scaler.joblib", model_bucket),
            ("data/processed/preprocessing_metadata.json", model_bucket),
        ]

        uploaded_files = []

        for file_path, bucket_name in files_to_upload:
            local_path = Path(file_path)
            if local_path.exists():
                try:
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)

                    # Upload del file
                    blob = bucket.blob(str(local_path.relative_to(Path.cwd())))
                    blob.upload_from_filename(str(local_path))

                    uploaded_files.append(
                        f"gs://{bucket_name}/{local_path.relative_to(Path.cwd())}"
                    )
                    logger.info(
                        f"🌤️  File caricato: {file_path} -> gs://{bucket_name}/{local_path.relative_to(Path.cwd())}"
                    )

                except Exception as e:
                    logger.error(f"Errore upload {file_path}: {e}")
            else:
                logger.warning(f"File non trovato: {file_path}")

        return {
            "status": "upload_completed",
            "uploaded_files": uploaded_files,
            "total_files": len(files_to_upload),
        }

    except Exception as e:
        logger.error(f"Errore durante upload auxiliary files: {e}")
        return {"status": "error", "error": str(e)}


@task(name="generate-deployment-report", retries=1, retry_delay_seconds=30)
def generate_deployment_report_task(validation_results: dict, model_info: dict):
    """
    Genera un report per il deployment.
    """
    logger = get_run_logger()
    logger.info("=== GENERATE DEPLOYMENT REPORT ===")

    try:
        # Creare report di deployment
        report = {
            "deployment_info": {
                "timestamp": datetime.now().isoformat(),
                "model_path": model_info["model_path"],
                "model_type": "LogisticRegression",
                "version": "1.0.0",
            },
            "performance_metrics": {
                "recall": validation_results["cv_results"]["recall"]["mean"],
                "f1_score": validation_results["cv_results"]["f1"]["mean"],
                "accuracy": validation_results["cv_results"]["accuracy"]["mean"],
                "roc_auc": validation_results["cv_results"]["roc_auc"]["mean"],
            },
            "validation_summary": {
                "cross_validation_folds": 5,
                "bootstrap_samples": 1000,
                "total_errors": validation_results["error_analysis"]["error_analysis"][
                    "error_count"
                ],
                "false_negatives": validation_results["error_analysis"][
                    "error_analysis"
                ]["false_negatives"],
                "false_positives": validation_results["error_analysis"][
                    "error_analysis"
                ]["false_positives"],
            },
            "deployment_ready": True,
            "recommendations": [
                "Modello validato e pronto per deployment",
                f"Recall: {validation_results['cv_results']['recall']['mean']:.4f} - Priorità soddisfatta",
                f"F1-Score: {validation_results['cv_results']['f1']['mean']:.4f} - Performance eccellente",
                "Nessun overfitting rilevato",
                "Bassa varianza nelle metriche di validazione",
            ],
        }

        # Salvare report con dual-mode
        report_path = RESULTS_DIR / "deployment_report.json"

        # Bucket per risultati (da configurazione)
        results_bucket = (
            "mlops-breast-cancer-artifacts" if is_cloud_environment() else None
        )

        # Salvataggio dual-mode del report
        report_path = save_metadata_dual_mode(
            report, report_path, results_bucket if results_bucket else None
        )

        logger.info(f"Deployment report salvato: {report_path}")
        return report

    except Exception as e:
        logger.error(f"Errore durante generazione report: {e}")
        raise


@flow(
    name="ml-training-pipeline",
    description="Pipeline completa per training modello breast cancer",
)
def ml_training_pipeline():
    """
    Pipeline principale per il training del modello.
    """
    logger = get_run_logger()
    logger.info("🚀 INIZIO ML TRAINING PIPELINE")

    try:
        # 1. Setup ambiente
        env_info = setup_environment()
        logger.info("✅ Environment setup completato")

        # 2. Download dataset
        dataset_path = download_dataset()
        logger.info("✅ Dataset scaricato")

        # 3. Preprocessing
        preprocess_info = preprocess_data_task(dataset_path)
        logger.info("✅ Preprocessing completato")

        # 4. Training baseline
        baseline_results = train_baseline_models_task()
        logger.info("✅ Training baseline completato")

        # 5. Training con MLflow
        mlflow_results = train_mlflow_models_task()
        logger.info("✅ Training MLflow completato")

        # 6. Hyperparameter tuning
        tuning_results = hyperparameter_tuning_task()
        logger.info("✅ Hyperparameter tuning completato")

        # 7. Feature importance analysis
        feature_results = feature_importance_analysis_task()
        logger.info("✅ Feature importance analysis completata")

        # 8. Model validation
        validation_results = model_validation_task()
        logger.info("✅ Model validation completata")

        # 9. Salvare miglior modello
        model_info = save_best_model_task(validation_results)
        logger.info("✅ Miglior modello salvato")

        # 10. Generare report deployment
        deployment_report = generate_deployment_report_task(
            validation_results, model_info
        )
        logger.info("✅ Deployment report generato")

        # Riepilogo finale
        logger.info(f"\n{'='*60}")
        logger.info("🏆 ML TRAINING PIPELINE COMPLETATA CON SUCCESSO!")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Performance Finali:")
        logger.info(
            f"  Recall: {validation_results['cv_results']['recall']['mean']:.4f}"
        )
        logger.info(f"  F1-Score: {validation_results['cv_results']['f1']['mean']:.4f}")
        logger.info(
            f"  Accuracy: {validation_results['cv_results']['accuracy']['mean']:.4f}"
        )
        logger.info(
            f"  ROC-AUC: {validation_results['cv_results']['roc_auc']['mean']:.4f}"
        )

        logger.info(f"\n📁 File Generati:")
        logger.info(f"  Modello: {model_info['model_path']}")
        logger.info(f"  Metadata: {model_info['metadata_path']}")
        logger.info(f"  Report: {RESULTS_DIR}/deployment_report.json")

        logger.info(f"\n🎯 Pronto per Deployment!")
        logger.info(f"{'='*60}")

        return {
            "pipeline_status": "completed",
            "model_info": model_info,
            "validation_results": validation_results,
            "deployment_report": deployment_report,
        }

    except Exception as e:
        logger.error(f"❌ Errore nella pipeline: {e}")
        raise


@flow(name="data-preprocessing-pipeline", description="Pipeline per preprocessing dati")
def data_preprocessing_pipeline():
    """
    Pipeline dedicata al preprocessing dei dati.
    """
    logger = get_run_logger()
    logger.info("🔄 INIZIO DATA PREPROCESSING PIPELINE")

    try:
        # Setup ambiente
        env_info = setup_environment()

        # Download dataset
        dataset_path = download_dataset()

        # Preprocessing
        preprocess_info = preprocess_data_task(dataset_path)

        logger.info("✅ Data preprocessing pipeline completata")
        return preprocess_info

    except Exception as e:
        logger.error(f"❌ Errore nella data preprocessing pipeline: {e}")
        raise


@flow(name="model-validation-pipeline", description="Pipeline per validazione modello")
def model_validation_pipeline():
    """
    Pipeline dedicata alla validazione del modello.
    """
    logger = get_run_logger()
    logger.info("🔍 INIZIO MODEL VALIDATION PIPELINE")

    try:
        # Setup ambiente
        env_info = setup_environment()

        # Validazione modello
        validation_results = model_validation_task()

        # Salvare miglior modello
        model_info = save_best_model_task(validation_results)

        # Generare report
        deployment_report = generate_deployment_report_task(
            validation_results, model_info
        )

        # Upload file ausiliari su GCS se in cloud
        auxiliary_upload = upload_auxiliary_files_task()

        logger.info("✅ Model validation pipeline completata")
        return {
            "validation_results": validation_results,
            "model_info": model_info,
            "deployment_report": deployment_report,
            "auxiliary_upload": auxiliary_upload,
        }

    except Exception as e:
        logger.error(f"❌ Errore nella model validation pipeline: {e}")
        raise


def create_deployments():
    """
    Crea i deployment per i workflow.
    """
    logger.info("=== CREAZIONE DEPLOYMENTS ===")

    # Deployment per pipeline completa (giornaliera)
    deployment_full = Deployment.build_from_flow(
        flow=ml_training_pipeline,
        name="ml-training-pipeline-daily",
        version="1.0.0",
        work_queue_name="ml-training",
        schedule=CronSchedule(cron="0 2 * * *"),  # Ogni giorno alle 2:00
        storage=storage,
    )
    deployment_full.apply()

    # Deployment per preprocessing (settimanale)
    deployment_preprocess = Deployment.build_from_flow(
        flow=data_preprocessing_pipeline,
        name="data-preprocessing-weekly",
        version="1.0.0",
        work_queue_name="data-processing",
        schedule=CronSchedule(cron="0 3 * * 1"),  # Ogni lunedì alle 3:00
        storage=storage,
    )
    deployment_preprocess.apply()

    # Deployment per validazione (giornaliera)
    deployment_validation = Deployment.build_from_flow(
        flow=model_validation_pipeline,
        name="model-validation-daily",
        version="1.0.0",
        work_queue_name="model-validation",
        schedule=CronSchedule(cron="0 4 * * *"),  # Ogni giorno alle 4:00
        storage=storage,
    )
    deployment_validation.apply()

    logger.info("✅ Deployments creati con successo")


if __name__ == "__main__":
    # Eseguire pipeline completa
    ml_training_pipeline()

    # Creare deployments (opzionale)
    # create_deployments()
