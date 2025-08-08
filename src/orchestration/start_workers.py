#!/usr/bin/env python3
"""
Script per avviare i worker Prefect.
"""

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Lista dei processi worker
worker_processes = []


def signal_handler(sig, frame):
    """
    Gestisce l'interruzione dei worker.
    """
    logger.info("\n🛑 Interruzione workers...")
    for queue_name, process in worker_processes:
        try:
            process.terminate()
            logger.info(f"Worker '{queue_name}' terminato")
        except:
            pass
    sys.exit(0)


def start_worker(queue_name):
    """
    Avvia un worker per una specifica work queue.
    """
    try:
        logger.info(f"🚀 Avviando worker per '{queue_name}'...")

        worker_process = subprocess.Popen(
            ["prefect", "worker", "start", "-p", queue_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Verificare che il worker si sia avviato
        time.sleep(3)
        if worker_process.poll() is None:
            logger.info(f"✅ Worker '{queue_name}' avviato con successo")
            return queue_name, worker_process
        else:
            logger.error(f"❌ Errore nell'avvio worker '{queue_name}'")
            return None

    except Exception as e:
        logger.error(f"Errore nell'avvio worker '{queue_name}': {e}")
        return None


def start_all_workers():
    """
    Avvia tutti i worker necessari.
    """
    logger.info("=== AVVIO WORKERS PREFECT ===")

    work_queues = ["ml-training", "data-processing", "model-validation"]

    for queue_name in work_queues:
        worker_info = start_worker(queue_name)
        if worker_info:
            worker_processes.append(worker_info)
        time.sleep(2)  # Pausa tra l'avvio dei worker

    logger.info(f"✅ {len(worker_processes)} workers avviati")
    return len(worker_processes) > 0


def monitor_workers():
    """
    Monitora lo stato dei worker.
    """
    logger.info("📊 Monitoraggio workers attivo...")
    logger.info("Premi Ctrl+C per terminare")

    try:
        while True:
            active_workers = 0
            for queue_name, process in worker_processes:
                if process.poll() is None:
                    active_workers += 1
                else:
                    logger.warning(
                        f"⚠️ Worker '{queue_name}' terminato inaspettatamente"
                    )

            if active_workers == 0:
                logger.error("❌ Tutti i workers sono terminati")
                break

            time.sleep(30)  # Controllo ogni 30 secondi

    except KeyboardInterrupt:
        logger.info("🛑 Interruzione manuale")


def main():
    """
    Funzione principale.
    """
    # Registrare handler per interruzione
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=== PREFECT WORKERS MANAGER ===")

    try:
        # Verificare che Prefect sia installato
        result = subprocess.run(
            ["prefect", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error("Prefect non installato. Installare con: pip install prefect")
            return

        # Avviare workers
        if start_all_workers():
            logger.info("\n🎉 WORKERS AVVIATI CON SUCCESSO!")
            logger.info("📊 Dashboard: http://localhost:4200")
            logger.info("🔧 Workers attivi per le seguenti queue:")
            for queue_name, _ in worker_processes:
                logger.info(f"  - {queue_name}")

            # Monitorare workers
            monitor_workers()
        else:
            logger.error("❌ Impossibile avviare i workers")

    except Exception as e:
        logger.error(f"Errore durante avvio workers: {e}")


if __name__ == "__main__":
    main()
