#!/usr/bin/env python3
"""
Script per avviare il server Prefect con API REST integrata.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# Importa l'API che abbiamo creato
from orchestration.prefect_api import app as api_app

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_prefect_server():
    """Avvia il server Prefect in un processo separato."""
    logger.info("🚀 Avvio server Prefect...")

    # Avvia il server Prefect
    process = subprocess.Popen(
        ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
    )

    return process


def start_api_server():
    """Avvia l'API REST su porta 8000."""
    logger.info("🚀 Avvio API REST...")

    # Modifica l'API per essere compatibile con Prefect server locale
    uvicorn.run(api_app, host="0.0.0.0", port=8000, log_level="info")


def signal_handler(signum, frame):
    """Gestisce i segnali per spegnimento pulito."""
    logger.info("🛑 Ricevuto segnale di terminazione")
    sys.exit(0)


def main():
    """Avvia sia Prefect server che API REST."""
    logger.info("🌟 AVVIO PREFECT SERVER + API REST")
    logger.info("=" * 50)

    # Registra gestori di segnale
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Imposta ambiente per Prefect locale nel container
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    os.environ["PREFECT_SERVER_API_HOST"] = "0.0.0.0"

    try:
        # Avvia Prefect server in un thread separato
        prefect_thread = threading.Thread(target=start_prefect_server, daemon=True)
        prefect_thread.start()

        # Attendi che Prefect server sia pronto
        logger.info("⏳ Attendo avvio Prefect server...")
        time.sleep(15)  # Tempo per l'inizializzazione

        # Verifica che Prefect server sia disponibile
        import requests

        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:4200/api/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Prefect server pronto!")
                    break
            except:
                pass

            if i == max_retries - 1:
                logger.error("❌ Prefect server non risponde")
                return 1

            time.sleep(2)

        # Avvia API REST (processo principale)
        logger.info("🚀 Avvio API REST su porta 8000...")
        start_api_server()

    except KeyboardInterrupt:
        logger.info("🛑 Interruzione da tastiera")
    except Exception as e:
        logger.error(f"❌ Errore: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
