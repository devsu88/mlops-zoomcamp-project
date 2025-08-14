#!/usr/bin/env python3
"""
Configurazione MLflow per experiment tracking e model registry.
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione MLflow
# Dual-mode: locale vs cloud
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # local, cloud

if ENVIRONMENT == "cloud":
    MLFLOW_TRACKING_URI = "https://mlflow-server-403815755558.europe-west1.run.app"
    MLFLOW_REGISTRY_URI = "https://mlflow-server-403815755558.europe-west1.run.app"
else:
    MLFLOW_TRACKING_URI = "http://localhost:5000"  # Per sviluppo locale
    MLFLOW_REGISTRY_URI = "sqlite:///mlflow.db"  # Per sviluppo locale

EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_REGISTRY_NAME = "breast-cancer-model"


def setup_mlflow():
    """
    Configura MLflow per experiment tracking e model registry.
    """
    global ENVIRONMENT

    logger.info("=== SETUP MLFLOW ===")
    logger.info(f"Ambiente: {ENVIRONMENT}")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    try:
        # Configurare tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if ENVIRONMENT == "cloud":
            # Per cloud, verifichiamo la connessione
            try:
                mlflow.list_experiments()
                logger.info("✅ Connessione a MLflow Cloud stabilita!")
            except Exception as cloud_error:
                logger.error(f"Errore connessione MLflow Cloud: {cloud_error}")
                logger.info("Fallback a tracking locale...")
                mlflow.set_tracking_uri("file:./mlruns")
                ENVIRONMENT = "local"

        # Creare o ottenere experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Experiment creato: {EXPERIMENT_NAME} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Experiment esistente: {EXPERIMENT_NAME} (ID: {experiment_id})"
            )

        # Configurare experiment
        mlflow.set_experiment(EXPERIMENT_NAME)

        logger.info("MLflow configurato con successo!")
        return experiment_id

    except Exception as e:
        logger.error(f"Errore durante setup MLflow: {e}")
        logger.info("Continuando con tracking locale...")

        # Fallback a tracking locale
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(EXPERIMENT_NAME)
        return None


def log_model_params(model_params):
    """
    Logga i parametri del modello.
    """
    logger.info("Logging parametri modello...")
    mlflow.log_params(model_params)


def log_model_metrics(metrics):
    """
    Logga le metriche del modello.
    """
    logger.info("Logging metriche modello...")
    mlflow.log_metrics(metrics)


def log_model_artifacts(artifacts_path):
    """
    Logga gli artifacts (grafici, report, etc.).
    """
    logger.info(f"Logging artifacts da: {artifacts_path}")

    if Path(artifacts_path).exists():
        mlflow.log_artifacts(artifacts_path)
        logger.info("Artifacts loggati con successo!")
    else:
        logger.warning(f"Path artifacts non trovato: {artifacts_path}")


def log_model(model, model_name, model_type="sklearn"):
    """
    Logga il modello nel registry.
    """
    logger.info(f"Logging modello: {model_name}")

    try:
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        logger.info(f"Modello {model_name} loggato con successo!")

    except Exception as e:
        logger.error(f"Errore durante logging modello: {e}")


def register_model(model_name, model_version=None):
    """
    Registra il modello nel Model Registry.
    """
    logger.info(f"Registrazione modello: {model_name}")

    try:
        # Registrare modello
        model_details = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
            name=MODEL_REGISTRY_NAME,
        )

        logger.info(
            f"Modello registrato: {model_details.name} (v{model_details.version})"
        )
        return model_details

    except Exception as e:
        logger.error(f"Errore durante registrazione modello: {e}")
        return None


def get_model_from_registry(model_name, version=None):
    """
    Carica un modello dal Model Registry.
    """
    logger.info(f"Caricamento modello dal registry: {model_name}")

    try:
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Modello caricato: {model_uri}")
        return model

    except Exception as e:
        logger.error(f"Errore durante caricamento modello: {e}")
        return None


def log_experiment_info(experiment_info):
    """
    Logga informazioni aggiuntive sull'experiment.
    """
    logger.info("Logging informazioni experiment...")

    # Logga tags
    mlflow.set_tags(
        {
            "project": "breast-cancer-classification",
            "dataset": "breast-cancer-wisconsin",
            "task": "binary-classification",
            "target": "diagnosis",
            "features": "20",
            "cv_folds": "5",
        }
    )

    # Logga informazioni aggiuntive
    for key, value in experiment_info.items():
        mlflow.log_param(f"info_{key}", value)


def create_mlflow_run(
    model_name, model_params, metrics, artifacts_path=None, model=None
):
    """
    Crea una run MLflow completa.
    """
    logger.info(f"=== CREAZIONE RUN MLFLOW: {model_name} ===")

    with mlflow.start_run(run_name=f"{model_name}_baseline"):
        try:
            # Logga parametri
            log_model_params(model_params)

            # Logga metriche
            log_model_metrics(metrics)

            # Logga artifacts se disponibili
            if artifacts_path:
                log_model_artifacts(artifacts_path)

            # Logga informazioni experiment
            experiment_info = {
                "model_name": model_name,
                "dataset_size": "455 train, 114 test",
                "feature_selection": "SelectKBest (k=20)",
                "preprocessing": "StandardScaler",
            }
            log_experiment_info(experiment_info)

            # Logga modello se fornito
            if model is not None:
                model_type = "xgboost" if "xgboost" in model_name else "sklearn"
                log_model(model, model_name, model_type)

                # Registra modello
                model_details = register_model(model_name)
            else:
                model_details = None

            logger.info(f"Run MLflow completata: {mlflow.active_run().info.run_id}")
            return model_details

        except Exception as e:
            logger.error(f"Errore durante creazione run MLflow: {e}")
            return None


def get_best_model_from_registry():
    """
    Ottiene il miglior modello dal registry basato su metriche.
    """
    logger.info("Ricerca miglior modello dal registry...")

    try:
        # Per ora, restituisce l'ultima versione
        # In futuro, implementare logica per selezionare il migliore
        model = get_model_from_registry(MODEL_REGISTRY_NAME)
        return model

    except Exception as e:
        logger.error(f"Errore durante ricerca miglior modello: {e}")
        return None


def list_experiments():
    """
    Lista tutti gli experiments disponibili.
    """
    logger.info("=== LISTA EXPERIMENTS ===")

    try:
        experiments = mlflow.search_experiments()
        for exp in experiments:
            logger.info(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

        return experiments

    except Exception as e:
        logger.error(f"Errore durante lista experiments: {e}")
        return []


def list_model_versions():
    """
    Lista tutte le versioni del modello nel registry.
    """
    logger.info("=== LISTA VERSIONI MODELLO ===")

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        for version in versions:
            logger.info(f"Versione: {version.version} - Run ID: {version.run_id}")

        return versions

    except Exception as e:
        logger.error(f"Errore durante lista versioni: {e}")
        return []


if __name__ == "__main__":
    # Test configurazione
    setup_mlflow()
    list_experiments()
    list_model_versions()
