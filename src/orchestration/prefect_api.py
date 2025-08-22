#!/usr/bin/env python3
"""
API REST per automatizzare il setup e l'avvio della pipeline Prefect.
"""

import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Importa le funzioni dal modulo prefect_setup
from orchestration.prefect_setup import (
    ensure_work_queues,
    register_deployments,
    start_workers,
    trigger_deployment_by_name,
    wait_for_server_ready,
)
from orchestration.prefect_config import get_prefect_config, get_work_queues

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza FastAPI
app = FastAPI(
    title="MLOps Pipeline API",
    description="API per automatizzare il setup e l'avvio della pipeline MLOps",
    version="1.0.0",
)


# Modelli Pydantic per le richieste
class PipelineRequest(BaseModel):
    """Richiesta per l'avvio della pipeline."""

    environment: str = Field(default="cloud", description="Ambiente: local o cloud")
    deployment_name: str = Field(
        default="ml-training-pipeline", description="Nome del deployment da triggerare"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Parametri per la pipeline"
    )
    auto_start_worker: bool = Field(
        default=True, description="Avvia automaticamente un worker"
    )


class PipelineResponse(BaseModel):
    """Risposta dell'API."""

    success: bool
    message: str
    flow_run_id: Optional[str] = None
    dashboard_url: Optional[str] = None
    worker_pid: Optional[int] = None


class HealthResponse(BaseModel):
    """Risposta per health check."""

    status: str
    timestamp: str
    environment: str
    prefect_server: str


# Variabili globali per tracciare lo stato
worker_processes: Dict[str, subprocess.Popen] = {}
pipeline_status: Dict[str, Any] = {}


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint root con informazioni sull'API."""
    return {
        "message": "MLOps Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "pipeline": "/pipeline/start",
            "status": "/pipeline/status",
            "workers": "/workers/status",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check dell'API e del server Prefect."""
    try:
        config = get_prefect_config()
        server_status = "healthy"

        # Verifica connessione al server Prefect
        if not wait_for_server_ready(config["api_url"], timeout_s=10):
            server_status = "unhealthy"

        return HealthResponse(
            status="healthy",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            environment=config["environment"],
            prefect_server=server_status,
        )
    except Exception as e:
        logger.error(f"Health check fallito: {e}")
        raise HTTPException(status_code=500, detail=f"Health check fallito: {e}")


@app.post("/pipeline/start", response_model=PipelineResponse)
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Avvia la pipeline MLOps completa."""
    try:
        logger.info(f"🚀 Richiesta avvio pipeline: {request}")

        # 1. Imposta ambiente
        os.environ["ENVIRONMENT"] = request.environment
        os.environ[
            "PYTHONPATH"
        ] = f"{os.getcwd()}/src:{os.environ.get('PYTHONPATH', '')}"

        config = get_prefect_config()
        logger.info(f"Configurazione: {config}")

        # 2. Verifica server Prefect
        if not wait_for_server_ready(config["api_url"]):
            raise HTTPException(
                status_code=503, detail="Server Prefect non disponibile"
            )

        # 3. Crea work queues
        logger.info("Creazione work queues...")
        ensure_work_queues(get_work_queues())

        # 4. Registra deployments
        logger.info("Registrazione deployments...")
        register_deployments(api_url=config["api_url"], environment=request.environment)

        # 5. Avvia worker se richiesto
        worker_pid = None
        if request.auto_start_worker:
            logger.info("Avvio worker...")
            worker_pid = start_worker_background(
                queue_names=["ml-training"],
                api_url=config["api_url"],
                environment=request.environment,
            )

        # 6. Trigger pipeline
        logger.info(f"Trigger pipeline: {request.deployment_name}")
        flow_run_id = trigger_deployment_by_name(
            api_url=config["api_url"],
            deployment_name=request.deployment_name,
            parameters=request.parameters,
        )

        if not flow_run_id:
            raise HTTPException(
                status_code=500, detail="Impossibile triggerare la pipeline"
            )

        # 7. Salva stato
        pipeline_status[flow_run_id] = {
            "deployment_name": request.deployment_name,
            "environment": request.environment,
            "start_time": time.time(),
            "worker_pid": worker_pid,
            "status": "running",
        }

        logger.info(f"✅ Pipeline avviata con successo: {flow_run_id}")

        return PipelineResponse(
            success=True,
            message="Pipeline avviata con successo",
            flow_run_id=flow_run_id,
            dashboard_url=config["dashboard_url"],
            worker_pid=worker_pid,
        )

    except Exception as e:
        logger.error(f"❌ Errore nell'avvio della pipeline: {e}")
        raise HTTPException(
            status_code=500, detail=f"Errore nell'avvio della pipeline: {e}"
        )


@app.get("/pipeline/status/{flow_run_id}")
async def get_pipeline_status(flow_run_id: str):
    """Ottiene lo stato di una pipeline specifica."""
    if flow_run_id not in pipeline_status:
        raise HTTPException(status_code=404, detail="Pipeline non trovata")

    return pipeline_status[flow_run_id]


@app.get("/workers/status")
async def get_workers_status():
    """Ottiene lo stato di tutti i worker attivi."""
    workers_info = {}
    for queue_name, process in worker_processes.items():
        workers_info[queue_name] = {
            "pid": process.pid,
            "alive": process.poll() is None,
            "returncode": process.returncode,
        }

    return {"active_workers": len(worker_processes), "workers": workers_info}


@app.delete("/workers/stop/{queue_name}")
async def stop_worker(queue_name: str):
    """Ferma un worker specifico."""
    if queue_name not in worker_processes:
        raise HTTPException(status_code=404, detail="Worker non trovato")

    try:
        process = worker_processes[queue_name]
        process.terminate()
        process.wait(timeout=10)
        del worker_processes[queue_name]

        return {"message": f"Worker {queue_name} fermato con successo"}
    except Exception as e:
        logger.error(f"Errore nel fermare worker {queue_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel fermare worker: {e}")


def start_worker_background(
    queue_names: List[str], api_url: str, environment: str = "cloud"
) -> Optional[int]:
    """Avvia un worker in background e restituisce il PID."""
    try:
        for queue_name in queue_names:
            env = os.environ.copy()
            env["PREFECT_API_URL"] = api_url
            env["ENVIRONMENT"] = environment
            env.setdefault("PROJECT_ROOT", os.getcwd())

            current_src = os.path.join(os.getcwd(), "src")
            if "PYTHONPATH" in env and env["PYTHONPATH"]:
                if current_src not in env["PYTHONPATH"].split(":"):
                    env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{current_src}"
            else:
                env["PYTHONPATH"] = current_src

            logger.info(f"🚀 Avvio worker per '{queue_name}' su {api_url}")

            process = subprocess.Popen(
                [
                    "prefect",
                    "worker",
                    "start",
                    "--pool",
                    "default-agent-pool",
                    "--work-queue",
                    queue_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            worker_processes[queue_name] = process
            time.sleep(2)

            return process.pid

        # Se non ci sono queue_names, ritorna None
        return None

    except Exception as e:
        logger.error(f"Errore nell'avvio worker: {e}")
        return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
