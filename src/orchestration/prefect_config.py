#!/usr/bin/env python3
"""
Configurazione dual-mode per Prefect (locale vs cloud).
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
    PREFECT_SERVER_URI = "https://prefect-server-403815755558.europe-west1.run.app"
    PREFECT_API_URL = "https://prefect-server-403815755558.europe-west1.run.app/api"
    PREFECT_DASHBOARD_URL = "https://prefect-server-403815755558.europe-west1.run.app"

    # Database cloud (SQLite nel container)
    PREFECT_DATABASE_URL = os.getenv(
        "PREFECT_DATABASE_URL",
        "sqlite:///prefect.db",
    )

    logger.info("🌤️  Configurazione Prefect: AMBIENTE CLOUD")
else:
    # Configurazione per ambiente locale
    PREFECT_SERVER_URI = "http://localhost:4200"
    PREFECT_API_URL = "http://localhost:4200/api"
    PREFECT_DASHBOARD_URL = "http://localhost:4200"

    # Database locale (SQLite)
    PREFECT_DATABASE_URL = "sqlite:///prefect.db"

    logger.info("🏠  Configurazione Prefect: AMBIENTE LOCALE")

# Configurazioni comuni
PREFECT_PROJECT_NAME = "breast-cancer-mlops"
PREFECT_WORK_QUEUES = ["ml-training", "data-processing", "model-validation"]


def get_prefect_config():
    """
    Restituisce la configurazione Prefect per l'ambiente corrente.
    """
    config = {
        "environment": ENVIRONMENT,
        "server_uri": PREFECT_SERVER_URI,
        "api_url": PREFECT_API_URL,
        "dashboard_url": PREFECT_DASHBOARD_URL,
        "database_url": PREFECT_DATABASE_URL,
        "project_name": PREFECT_PROJECT_NAME,
        "work_queues": PREFECT_WORK_QUEUES,
    }

    logger.info(f"Configurazione Prefect: {config}")
    return config


def is_cloud_environment():
    """
    Verifica se siamo in ambiente cloud.
    """
    return ENVIRONMENT == "cloud"


def get_database_url():
    """
    Restituisce l'URL del database per l'ambiente corrente.
    """
    return PREFECT_DATABASE_URL


def get_work_queues():
    """
    Restituisce la lista delle work queue.
    """
    return PREFECT_WORK_QUEUES.copy()
