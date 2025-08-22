#!/usr/bin/env python3
"""
API integrata per triggerare pipeline tramite endpoint Prefect.
"""

import os
import requests
import time
from typing import Dict, Any, Optional

from orchestration.prefect_setup import (
    ensure_work_queues,
    register_deployments,
    trigger_deployment_by_name,
    wait_for_server_ready,
)
from orchestration.prefect_config import get_prefect_config, get_work_queues


def trigger_pipeline_via_api(
    environment: str = "cloud",
    deployment_name: str = "ml-training-pipeline",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Triggera la pipeline tramite API Prefect.
    """
    try:
        # 1. Imposta ambiente
        os.environ["ENVIRONMENT"] = environment
        os.environ[
            "PYTHONPATH"
        ] = f"{os.getcwd()}/src:{os.environ.get('PYTHONPATH', '')}"

        config = get_prefect_config()

        # 2. Verifica server Prefect
        if not wait_for_server_ready(config["api_url"]):
            return {"success": False, "error": "Server Prefect non disponibile"}

        # 3. Crea work queues
        ensure_work_queues(get_work_queues())

        # 4. Registra deployments
        register_deployments(api_url=config["api_url"], environment=environment)

        # 5. Trigger pipeline
        flow_run_id = trigger_deployment_by_name(
            api_url=config["api_url"],
            deployment_name=deployment_name,
            parameters=parameters or {},
        )

        if not flow_run_id:
            return {"success": False, "error": "Impossibile triggerare la pipeline"}

        return {
            "success": True,
            "flow_run_id": flow_run_id,
            "dashboard_url": config["dashboard_url"],
            "message": "Pipeline avviata con successo",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_pipeline_status(flow_run_id: str) -> Dict[str, Any]:
    """
    Ottiene lo stato di una pipeline.
    """
    try:
        config = get_prefect_config()

        # Query Prefect API per lo stato del flow run
        response = requests.get(
            f"{config['api_url']}/flow_runs/{flow_run_id}", timeout=10
        )

        if response.status_code == 200:
            flow_run = response.json()
            return {
                "success": True,
                "flow_run_id": flow_run_id,
                "state": flow_run.get("state", {}).get("name", "unknown"),
                "start_time": flow_run.get("start_time"),
                "end_time": flow_run.get("end_time"),
            }
        else:
            return {
                "success": False,
                "error": f"Impossibile ottenere stato: {response.status_code}",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test della funzione
    result = trigger_pipeline_via_api()
    print(f"Risultato: {result}")
