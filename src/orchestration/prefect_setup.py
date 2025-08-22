#!/usr/bin/env python3
"""
Setup centralizzato per Prefect (dual-mode: locale/cloud).
Avvio server (solo locale), creazione work queues, registrazione deployment, avvio worker e trigger manuale.
"""

import argparse
import json
import logging
import os
import subprocess
import time
from typing import List, Optional

import requests
from prefect.settings import (
    temporary_settings,
    PREFECT_API_URL as PREFECT_API_URL_SETTING,
)

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config dual-mode
from orchestration.prefect_config import (
    get_prefect_config,
    is_cloud_environment,
    get_work_queues,
)


def start_server_local() -> bool:
    """Avvia il server Prefect in locale."""
    logger.info("=== AVVIO PREFECT SERVER (LOCALE) ===")
    try:
        result = subprocess.run(
            ["prefect", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error("Prefect non installato. Installare con: pip install prefect")
            return False

        server_process = subprocess.Popen(
            ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(5)
        if server_process.poll() is None:
            logger.info("✅ Prefect server avviato: http://localhost:4200")
            return True
        logger.error("❌ Errore nell'avvio del server Prefect")
        return False
    except Exception as exc:  # pragma: no cover
        logger.error(f"Errore durante avvio server: {exc}")
        return False


def wait_for_server_ready(api_url: str, timeout_s: int = 120) -> bool:
    """Attende che l'API del server risponda."""
    logger.info(f"⏳ Attendo disponibilità API: {api_url}")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_url}/health", timeout=3)
            if r.ok:
                logger.info("🟢 Prefect API disponibile")
                return True
        except Exception:
            pass
        time.sleep(2)
    logger.error("🔴 Prefect API non disponibile nei tempi previsti")
    return False


def ensure_work_queues(
    queue_names: List[str], pool_name: str = "default-agent-pool"
) -> None:
    """Crea le work queue se mancanti (idempotente)."""
    logger.info("=== CREAZIONE/VERIFICA WORK QUEUES ===")
    for queue_name in queue_names:
        try:
            result = subprocess.run(
                ["prefect", "work-queue", "create", "--pool", pool_name, queue_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"✅ Work queue '{queue_name}' creata")
            else:
                logger.info(
                    f"ℹ️ Work queue '{queue_name}' già esistente o non necessaria"
                )
        except Exception as exc:  # pragma: no cover
            logger.error(f"Errore nella creazione work queue '{queue_name}': {exc}")


def register_deployments(api_url: str, environment: str | None = None) -> None:
    """Registra i deployment (unico manuale) dai flow.

    Se `environment` è passato, forza ENVIRONMENT per scegliere lo storage corretto.
    """
    logger.info("=== REGISTRAZIONE DEPLOYMENT ===")
    if environment:
        os.environ["ENVIRONMENT"] = environment
    from orchestration.prefect_workflows import create_deployments

    # Forza l'uso dell'API corretta per la registrazione
    with temporary_settings({PREFECT_API_URL_SETTING: api_url}):
        create_deployments()
    logger.info("✅ Deployment registrato")


def start_workers(
    queue_names: List[str],
    api_url: str,
    environment: str = "local",
    pool_name: str = "default-agent-pool",
) -> None:
    """Avvia i worker per le work queue specificate puntando a PREFECT_API_URL.

    Propaga ENVIRONMENT (local/cloud) e PYTHONPATH=./src per garantire gli import.
    """
    logger.info("=== AVVIO WORKERS ===")
    for queue_name in queue_names:
        try:
            env = os.environ.copy()
            env["PREFECT_API_URL"] = api_url
            env["ENVIRONMENT"] = environment
            # Base del progetto per scrivere/leggere dati locali nella cartella data/
            env.setdefault("PROJECT_ROOT", os.getcwd())
            # Garantisce import dei moduli di progetto
            current_src = os.path.join(os.getcwd(), "src")
            if "PYTHONPATH" in env and env["PYTHONPATH"]:
                if current_src not in env["PYTHONPATH"].split(":"):
                    env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{current_src}"
            else:
                env["PYTHONPATH"] = current_src
            logger.info(f"🚀 Avvio worker per '{queue_name}' su {api_url}")
            subprocess.Popen(
                [
                    "prefect",
                    "worker",
                    "start",
                    "--pool",
                    pool_name,
                    "--work-queue",
                    queue_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            time.sleep(2)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Errore nell'avvio worker '{queue_name}': {exc}")
    logger.info("✅ Richiesto avvio workers")


def trigger_deployment_by_name(
    api_url: str, deployment_name: str, parameters: Optional[dict] = None
) -> Optional[str]:
    """Risolvi il deployment per nome e crea un flow run (Prefect 2 API)."""
    try:
        # Prefect 2 richiede POST su /deployments/filter (GET porta a 405)
        rf = requests.post(f"{api_url}/deployments/filter", json={}, timeout=10)
        rf.raise_for_status()
        deployments = (
            rf.json() if isinstance(rf.json(), list) else rf.json().get("items", [])
        )

        deployment_id = None
        for d in deployments:
            if d.get("name") == deployment_name:
                deployment_id = d.get("id")
                break
        if not deployment_id:
            logger.error(f"Deployment non trovato: {deployment_name}")
            return None

        payload = {"parameters": parameters or {}}
        rc = requests.post(
            f"{api_url}/deployments/{deployment_id}/create_flow_run",
            json=payload,
            timeout=15,
        )
        if rc.ok:
            flow_run_id = rc.json().get("id")
            logger.info(f"✅ Flow run creato: {flow_run_id}")
            return flow_run_id
        logger.error(f"Errore trigger deployment: {rc.status_code} {rc.text}")
        return None
    except Exception as exc:  # pragma: no cover
        logger.error(f"Errore chiamata API Prefect: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Prefect (dual-mode)")
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument(
        "--actions",
        default="queues,deployments",
        help="Sequenza di azioni: server,queues,deployments,workers,run",
    )
    parser.add_argument("--deployment", default="ml-training-pipeline")
    args = parser.parse_args()

    cfg = get_prefect_config()
    api_url = cfg["api_url"]

    logger.info(f"🌐 Modalità: {args.mode} | API: {api_url}")

    steps = [s.strip() for s in args.actions.split(",") if s.strip()]

    if args.mode == "local":
        if "server" in steps:
            if not start_server_local():
                raise SystemExit(1)
        if not wait_for_server_ready(api_url):
            raise SystemExit(1)
    else:
        # Cloud: assumiamo server gestito e raggiungibile
        if not wait_for_server_ready(api_url):
            raise SystemExit(1)

    queues = get_work_queues()

    if "queues" in steps:
        ensure_work_queues(queues)

    if "deployments" in steps:
        register_deployments(api_url=api_url, environment=args.mode)

    if "workers" in steps:
        start_workers(
            queue_names=["ml-training"], api_url=api_url, environment=args.mode
        )

    if "run" in steps:
        trigger_deployment_by_name(api_url=api_url, deployment_name=args.deployment)


if __name__ == "__main__":
    main()
