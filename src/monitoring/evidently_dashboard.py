"""
Evidently Dashboard per il monitoring
Dashboard web per visualizzare tutti i report di monitoring
"""
import json
import logging
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)

# Evidently 0.4.0 non ha il dashboard integrato
# Useremo i report HTML invece
from evidently.report import Report

from src.monitoring.monitoring_config import (
    EVIDENTLY_HOST,
    EVIDENTLY_PORT,
    FEATURE_CONFIG,
    ensure_directories,
    get_monitoring_paths,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidentlyDashboard:
    """Dashboard Evidently per il monitoring"""

    def __init__(self):
        self.paths = get_monitoring_paths()
        ensure_directories()
        self.column_mapping = ColumnMapping(
            target=FEATURE_CONFIG["target_column"],
            numerical_features=FEATURE_CONFIG["numerical_features"],
            categorical_features=FEATURE_CONFIG["categorical_features"],
        )
        self.dashboard = None

    def load_data_for_dashboard(self):
        """Carica i dati necessari per il dashboard"""
        try:
            # Carica dati di riferimento
            reference_data_path = Path("data/processed/train_set.csv")
            reference_data = pd.read_csv(reference_data_path)
            logger.info(f"✓ Dati di riferimento caricati: {reference_data.shape}")

            # Carica dati attuali (per ora usa test set)
            current_data_path = Path("data/processed/test_set.csv")
            current_data = pd.read_csv(current_data_path)
            logger.info(f"✓ Dati attuali caricati: {current_data.shape}")

            # Carica modello per generare predizioni
            model_path = Path(
                "mlruns/345803254500172789/c0edfdfcb43c46758634223da7d1faa5/artifacts/logistic_regression/model.pkl"
            )
            scaler_path = Path("data/processed/scaler.joblib")

            # Per ora usiamo i dati senza predizioni per evitare problemi con Evidently
            logger.info(f"✓ Usando dati senza predizioni per dashboard")
            return reference_data, current_data

        except Exception as e:
            logger.error(f"❌ Errore nel caricamento dati per dashboard: {e}")
            return None, None

    def create_dashboard(self, reference_data, current_data):
        """Crea il dashboard Evidently (versione semplificata)"""
        try:
            # Crea report solo per data quality e drift (senza classification)
            comprehensive_report = Report(
                metrics=[DataQualityPreset(), DataDriftPreset()]
            )

            comprehensive_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Salva dashboard come report HTML
            dashboard_path = self.paths["dashboard"]
            comprehensive_report.save_html(dashboard_path)
            logger.info(f"✓ Dashboard salvato: {dashboard_path}")

            return dashboard_path

        except Exception as e:
            logger.error(f"❌ Errore nella creazione dashboard: {e}")
            return None

    def serve_dashboard(self, dashboard_path):
        """Apri il dashboard nel browser"""
        try:
            # Apri il file HTML nel browser
            import os
            import webbrowser

            # Converti path relativo in assoluto
            abs_path = os.path.abspath(dashboard_path)
            file_url = f"file://{abs_path}"

            logger.info(f"✓ Apertura dashboard: {file_url}")
            webbrowser.open(file_url)

            return True

        except Exception as e:
            logger.error(f"❌ Errore nell'apertura dashboard: {e}")
            return False

    def create_comprehensive_report(self, reference_data, current_data):
        """Crea un report completo con tutte le metriche"""
        try:
            # Crea report con metriche di data quality e drift (senza classification)
            comprehensive_report = Report(
                metrics=[DataQualityPreset(), DataDriftPreset()]
            )

            comprehensive_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Salva report
            report_path = (
                self.paths["reports_dir"] / "comprehensive_monitoring_report.html"
            )
            comprehensive_report.save_html(report_path)
            logger.info(f"✓ Report completo salvato: {report_path}")

            return report_path

        except Exception as e:
            logger.error(f"❌ Errore nella creazione report completo: {e}")
            return None

    def generate_monitoring_summary(self):
        """Genera un summary di tutti i report di monitoring"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "monitoring_status": "active",
                "reports": {},
                "alerts": [],
                "summary": {},
            }

            # Controlla report esistenti
            reports_dir = self.paths["reports_dir"]

            # Data Quality
            data_quality_path = reports_dir / "data_quality_metrics.json"
            if data_quality_path.exists():
                with open(data_quality_path, "r") as f:
                    data_quality_metrics = json.load(f)
                summary["reports"]["data_quality"] = data_quality_metrics

                # Controlla alert
                if data_quality_metrics.get("data_quality_score", 1.0) < 0.8:
                    summary["alerts"].append("Data quality score basso")

            # Data Drift
            drift_path = reports_dir / "drift_metrics.json"
            if drift_path.exists():
                with open(drift_path, "r") as f:
                    drift_metrics = json.load(f)
                summary["reports"]["data_drift"] = drift_metrics

                # Controlla alert
                if drift_metrics.get("dataset_drift_detected", False):
                    summary["alerts"].append("Dataset drift rilevato")

            # Performance
            performance_path = reports_dir / "performance_metrics.json"
            if performance_path.exists():
                with open(performance_path, "r") as f:
                    performance_metrics = json.load(f)
                summary["reports"]["model_performance"] = performance_metrics

                # Controlla alert
                current_perf = performance_metrics.get("current_performance", {})
                if current_perf.get("recall", 1.0) < 0.9:
                    summary["alerts"].append("Recall del modello bassa")

            # Conta alert totali
            summary["summary"]["total_alerts"] = len(summary["alerts"])
            summary["summary"]["total_reports"] = len(summary["reports"])

            # Salva summary
            summary_path = self.paths["reports_dir"] / "monitoring_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"✓ Summary monitoring salvato: {summary_path}")
            return summary

        except Exception as e:
            logger.error(f"❌ Errore nella generazione summary: {e}")
            return None

    def start_monitoring_dashboard(self):
        """Avvia il dashboard di monitoring completo"""
        print("🚀 Avvio Evidently Dashboard...")

        # Carica dati
        reference_data, current_data = self.load_data_for_dashboard()

        if reference_data is not None and current_data is not None:
            # Crea dashboard
            dashboard_path = self.create_dashboard(reference_data, current_data)

            if dashboard_path:
                # Crea report completo
                comprehensive_report_path = self.create_comprehensive_report(
                    reference_data, current_data
                )

                # Genera summary
                summary = self.generate_monitoring_summary()

                print(f"\n📊 Evidently Dashboard")
                print(f"Dashboard: {dashboard_path}")
                if comprehensive_report_path:
                    print(f"Report completo: {comprehensive_report_path}")

                if summary:
                    print(f"Alert totali: {summary['summary']['total_alerts']}")
                    if summary["alerts"]:
                        print(f"Alert attivi:")
                        for alert in summary["alerts"]:
                            print(f"  - {alert}")

                # Apri dashboard
                print(f"\n🌐 Apertura dashboard...")
                success = self.serve_dashboard(dashboard_path)

                if success:
                    print(f"✅ Dashboard aperto nel browser")
                    print(f"📱 Il file HTML è stato aperto automaticamente")
                else:
                    print(f"❌ Errore nell'apertura del dashboard")
            else:
                print("❌ Errore nella creazione del dashboard")
        else:
            print("❌ Impossibile caricare i dati per il dashboard")


def main():
    """Funzione principale per il dashboard"""
    dashboard = EvidentlyDashboard()
    dashboard.start_monitoring_dashboard()


if __name__ == "__main__":
    main()
