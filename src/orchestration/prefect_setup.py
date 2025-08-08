#!/usr/bin/env python3
"""
Script per configurare il server Prefect e i work queue.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def start_prefect_server():
    """
    Avvia il server Prefect.
    """
    logger.info("=== AVVIO PREFECT SERVER ===")

    try:
        # Verificare se Prefect è installato
        result = subprocess.run(
            ["prefect", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error("Prefect non installato. Installare con: pip install prefect")
            return False

        # Avviare il server Prefect
        logger.info("Avviando Prefect server...")
        server_process = subprocess.Popen(
            ["prefect", "server", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Attendere che il server si avvii
        time.sleep(5)

        if server_process.poll() is None:
            logger.info("✅ Prefect server avviato con successo")
            logger.info("Dashboard disponibile su: http://localhost:4200")
            return True
        else:
            logger.error("❌ Errore nell'avvio del server Prefect")
            return False

    except Exception as e:
        logger.error(f"Errore durante avvio server: {e}")
        return False


def create_work_queues():
    """
    Crea i work queue necessari.
    """
    logger.info("=== CREAZIONE WORK QUEUES ===")

    work_queues = ["ml-training", "data-processing", "model-validation"]

    for queue_name in work_queues:
        try:
            result = subprocess.run(
                ["prefect", "work-queue", "create", queue_name],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"✅ Work queue '{queue_name}' creata")
            else:
                logger.info(f"ℹ️ Work queue '{queue_name}' già esistente")

        except Exception as e:
            logger.error(f"Errore nella creazione work queue '{queue_name}': {e}")


def start_workers():
    """
    Avvia i worker per i work queue.
    """
    logger.info("=== AVVIO WORKERS ===")

    work_queues = ["ml-training", "data-processing", "model-validation"]

    worker_processes = []

    for queue_name in work_queues:
        try:
            logger.info(f"Avviando worker per '{queue_name}'...")
            worker_process = subprocess.Popen(
                ["prefect", "worker", "start", "-p", queue_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            worker_processes.append((queue_name, worker_process))

            # Attendere un po' tra l'avvio dei worker
            time.sleep(2)

        except Exception as e:
            logger.error(f"Errore nell'avvio worker '{queue_name}': {e}")

    logger.info(f"✅ {len(worker_processes)} workers avviati")
    return worker_processes


def check_prefect_status():
    """
    Verifica lo stato di Prefect.
    """
    logger.info("=== VERIFICA STATO PREFECT ===")

    try:
        # Verificare server
        result = subprocess.run(
            ["prefect", "server", "status"], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("✅ Prefect server attivo")
        else:
            logger.warning("⚠️ Prefect server non attivo")

        # Verificare work queues
        result = subprocess.run(
            ["prefect", "work-queue", "ls"], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("✅ Work queues disponibili")
            logger.info(result.stdout)
        else:
            logger.warning("⚠️ Errore nel controllo work queues")

    except Exception as e:
        logger.error(f"Errore nella verifica stato: {e}")


def setup_prefect_environment():
    """
    Setup completo dell'ambiente Prefect.
    """
    logger.info("🚀 SETUP AMBIENTE PREFECT")

    # 1. Avviare server
    if not start_prefect_server():
        logger.error("❌ Impossibile avviare il server Prefect")
        return False

    # 2. Attendere che il server sia pronto
    time.sleep(10)

    # 3. Creare work queues
    create_work_queues()

    # 4. Verificare stato
    check_prefect_status()

    logger.info("✅ Setup ambiente Prefect completato")
    return True


def main():
    """
    Funzione principale.
    """
    logger.info("=== SETUP PREFECT ORCHESTRATION ===")

    try:
        # Setup ambiente
        if setup_prefect_environment():
            logger.info("\n🎉 PREFECT SETUP COMPLETATO!")
            logger.info("📊 Dashboard: http://localhost:4200")
            logger.info(
                "🔧 Per avviare workers: python src/orchestration/start_workers.py"
            )
            logger.info(
                "🚀 Per eseguire pipeline: python src/orchestration/prefect_workflows.py"
            )
        else:
            logger.error("❌ Setup Prefect fallito")

    except Exception as e:
        logger.error(f"Errore durante setup: {e}")


if __name__ == "__main__":
    main()
