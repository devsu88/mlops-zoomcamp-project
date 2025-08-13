#!/usr/bin/env python3
"""
Wrapper Flask per esporre la funzione batch_monitoring come HTTP endpoint.
Questo permette di usare la stessa logica dual-mode in Cloud Run.
"""

import os
import logging
from flask import Flask, request, jsonify
from monitoring.batch_monitoring import run_monitoring

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "batch-monitoring",
            "environment": os.getenv("ENVIRONMENT", "local"),
        }
    )


@app.route("/monitor", methods=["POST"])
def monitor():
    """Endpoint principale per eseguire il batch monitoring."""
    try:
        logger.info("🚀 Avvio batch monitoring...")

        # Crea un mock request object per compatibilità
        class MockRequest:
            def get_json(self, silent=True):
                return request.get_json(silent=silent) if request.is_json else None

        # Esegui il monitoring
        results = run_monitoring(MockRequest())

        logger.info("✅ Batch monitoring completato")
        return jsonify(
            {
                "status": "success",
                "results": results,
                "environment": os.getenv("ENVIRONMENT", "local"),
            }
        )

    except Exception as e:
        logger.error(f"❌ Errore durante batch monitoring: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "environment": os.getenv("ENVIRONMENT", "local"),
                }
            ),
            500,
        )


@app.route("/monitor", methods=["GET"])
def monitor_get():
    """Endpoint GET per compatibilità."""
    return jsonify(
        {
            "message": "Batch Monitoring Service",
            "usage": "POST /monitor per eseguire il monitoring",
            "environment": os.getenv("ENVIRONMENT", "local"),
        }
    )


if __name__ == "__main__":
    # Per sviluppo locale
    app.run(host="0.0.0.0", port=8080, debug=False)
