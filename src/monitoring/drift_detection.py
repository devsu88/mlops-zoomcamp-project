"""
Data Drift Detection con Evidently AI
Rileva cambiamenti nella distribuzione dei dati rispetto al training set
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (ColumnDriftMetric, DataDriftTable,
                               DatasetDriftMetric)
from evidently.report import Report

from src.monitoring.monitoring_config import (DRIFT_THRESHOLDS, FEATURE_CONFIG,
                                              ensure_directories,
                                              get_monitoring_paths)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detector per il drift dei dati"""

    def __init__(self):
        self.paths = get_monitoring_paths()
        ensure_directories()
        self.column_mapping = ColumnMapping(
            target=FEATURE_CONFIG["target_column"],
            numerical_features=FEATURE_CONFIG["numerical_features"],
            categorical_features=FEATURE_CONFIG["categorical_features"],
        )

    def load_reference_data(self, data_path):
        """Carica i dati di riferimento (training set)"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✓ Dati di riferimento caricati: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Errore nel caricamento dati di riferimento: {e}")
            return None

    def load_current_data(self, data_path):
        """Carica i dati attuali (production data)"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✓ Dati attuali caricati: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Errore nel caricamento dati attuali: {e}")
            return None

    def generate_drift_report(self, reference_data, current_data):
        """Genera report di drift con Evidently"""
        try:
            # Genera report drift generale
            drift_report = Report(
                metrics=[DataDriftPreset(), DatasetDriftMetric(), DataDriftTable()]
            )

            drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Salva report HTML
            report_path = self.paths["drift_report"]
            drift_report.save_html(report_path)
            logger.info(f"✓ Report drift salvato: {report_path}")

            # Genera report drift per colonne specifiche
            column_drift_metrics = self._generate_column_drift_report(
                reference_data, current_data
            )

            # Estrai metriche chiave
            metrics = self._extract_drift_metrics(drift_report, column_drift_metrics)

            # Salva metriche JSON
            metrics_path = self.paths["reports_dir"] / "drift_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"✓ Metriche drift salvate: {metrics_path}")

            return metrics, report_path

        except Exception as e:
            logger.error(f"❌ Errore nella generazione report drift: {e}")
            return None, None

    def _generate_column_drift_report(self, reference_data, current_data):
        """Genera report di drift per colonne specifiche"""
        column_drift_metrics = {}

        for feature in FEATURE_CONFIG["numerical_features"]:
            try:
                # Crea report per singola colonna
                column_report = Report(metrics=[ColumnDriftMetric(column_name=feature)])

                column_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=self.column_mapping,
                )

                # Estrai metriche della colonna
                column_data = column_report.json()
                if "column_drift" in column_data:
                    drift_info = column_data["column_drift"]
                    column_drift_metrics[feature] = {
                        "drift_detected": drift_info.get("drift_detected", False),
                        "drift_score": drift_info.get("drift_score", 0.0),
                        "statistical_test": drift_info.get(
                            "statistical_test", "unknown"
                        ),
                        "p_value": drift_info.get("p_value", 1.0),
                    }

            except Exception as e:
                logger.warning(
                    f"⚠️ Errore nell'analisi drift per colonna {feature}: {e}"
                )
                column_drift_metrics[feature] = {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "statistical_test": "error",
                    "p_value": 1.0,
                }

        return column_drift_metrics

    def _extract_drift_metrics(self, drift_report, column_drift_metrics):
        """Estrae metriche chiave dal report Evidently"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift_detected": False,
            "drift_score": 0.0,
            "drifted_columns": [],
            "column_drift_details": column_drift_metrics,
            "summary": {},
        }

        try:
            # Estrai metriche dal report
            report_data = drift_report.json()

            # Debug: stampa il tipo e i primi caratteri
            logger.info(f"🔍 Tipo report_data: {type(report_data)}")
            if isinstance(report_data, str):
                logger.info(f"🔍 Primi 200 caratteri: {report_data[:200]}")

            # Verifica che report_data sia un dizionario
            if not isinstance(report_data, dict):
                # Prova a parsare come JSON se è una stringa
                if isinstance(report_data, str):
                    try:
                        import json

                        report_data = json.loads(report_data)
                        logger.info(
                            "✓ Report data parsato correttamente da stringa JSON"
                        )
                        logger.info(f"🔍 Chiavi disponibili: {list(report_data.keys())}")
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"❌ Impossibile parsare report data come JSON: {e}"
                        )
                        return metrics
                else:
                    logger.error(
                        f"❌ Report data non è né un dizionario né una stringa JSON: {type(report_data)}"
                    )
                    return metrics
            else:
                logger.info(f"🔍 Chiavi disponibili: {list(report_data.keys())}")

            # Estrai metriche dall'array metrics
            metrics_array = report_data.get("metrics", [])
            logger.info(f"🔍 Numero di metriche trovate: {len(metrics_array)}")

            for metric_item in metrics_array:
                if isinstance(metric_item, dict):
                    metric_name = metric_item.get("metric", "")
                    metric_result = metric_item.get("result", {})

                    logger.info(f"🔍 Processando metrica: {metric_name}")

                    if metric_name == "DatasetDriftMetric":
                        # Estrai informazioni sul dataset drift dalla struttura principale
                        logger.info(
                            f"🔍 DatasetDriftMetric - result keys: {list(metric_result.keys()) if isinstance(metric_result, dict) else 'not dict'}"
                        )

                        if isinstance(metric_result, dict):
                            # Estrai dalla struttura principale del result
                            drift_share = metric_result.get("drift_share", 0.0)
                            number_of_columns = metric_result.get(
                                "number_of_columns", 0
                            )
                            number_of_drifted_columns = metric_result.get(
                                "number_of_drifted_columns", 0
                            )
                            share_of_drifted_columns = metric_result.get(
                                "share_of_drifted_columns", 0.0
                            )

                            # Determina se c'è drift basandosi sulla percentuale
                            drift_detected = (
                                share_of_drifted_columns > 0.3
                            )  # Più del 30% delle colonne hanno drift

                            metrics["dataset_drift_detected"] = drift_detected
                            metrics["drift_score"] = share_of_drifted_columns
                            metrics["summary"]["total_columns"] = number_of_columns
                            metrics["summary"][
                                "drifted_columns"
                            ] = number_of_drifted_columns
                            metrics["summary"]["drift_share"] = share_of_drifted_columns

                            logger.info(
                                f"🔍 Drift detected: {drift_detected}, score: {share_of_drifted_columns:.3f}"
                            )
                            logger.info(
                                f"🔍 Total columns: {number_of_columns}, drifted: {number_of_drifted_columns}"
                            )

                    elif metric_name == "DataDriftTable":
                        # Estrai informazioni sulla tabella drift
                        logger.info(
                            f"🔍 DataDriftTable - result keys: {list(metric_result.keys()) if isinstance(metric_result, dict) else 'not dict'}"
                        )

                        if isinstance(metric_result, dict):
                            # Estrai dalla struttura principale del result
                            drift_share = metric_result.get("drift_share", 0.0)
                            number_of_columns = metric_result.get(
                                "number_of_columns", 0
                            )
                            number_of_drifted_columns = metric_result.get(
                                "number_of_drifted_columns", 0
                            )
                            share_of_drifted_columns = metric_result.get(
                                "share_of_drifted_columns", 0.0
                            )

                            metrics["summary"]["total_columns"] = number_of_columns
                            metrics["summary"][
                                "drifted_columns"
                            ] = number_of_drifted_columns
                            metrics["summary"]["drift_share"] = share_of_drifted_columns

                            logger.info(
                                f"🔍 Total columns: {number_of_columns}, drifted: {number_of_drifted_columns}"
                            )

                            # Estrai anche informazioni per colonna se disponibili
                            if "column_metrics" in metric_result:
                                column_metrics = metric_result["column_metrics"]
                                if isinstance(column_metrics, dict):
                                    for (
                                        column_name,
                                        column_data,
                                    ) in column_metrics.items():
                                        if isinstance(column_data, dict):
                                            drift_detected = column_data.get(
                                                "drift_detected", False
                                            )
                                            drift_score = column_data.get(
                                                "drift_score", 0.0
                                            )
                                            p_value = column_data.get("p_value", 1.0)

                                            if drift_detected:
                                                column_drift_metrics[column_name] = {
                                                    "drift_detected": drift_detected,
                                                    "drift_score": drift_score,
                                                    "p_value": p_value,
                                                }
                                                logger.info(
                                                    f"🔍 Drift rilevato per {column_name}: score={drift_score:.3f}, p_value={p_value:.3f}"
                                                )

            # Identifica colonne con drift
            drifted_columns = []
            for feature, drift_info in column_drift_metrics.items():
                if isinstance(drift_info, dict) and drift_info.get(
                    "drift_detected", False
                ):
                    drifted_columns.append(
                        {
                            "column": feature,
                            "drift_score": drift_info.get("drift_score", 0.0),
                            "p_value": drift_info.get("p_value", 1.0),
                        }
                    )

            metrics["drifted_columns"] = drifted_columns

            # Aggiungi summary
            metrics["summary"].update(
                {
                    "total_features": len(FEATURE_CONFIG["numerical_features"]),
                    "drifted_features_count": len(drifted_columns),
                    "drift_percentage": len(drifted_columns)
                    / len(FEATURE_CONFIG["numerical_features"])
                    if FEATURE_CONFIG["numerical_features"]
                    else 0,
                }
            )

        except Exception as e:
            logger.error(f"❌ Errore nell'estrazione metriche drift: {e}")
            metrics["issues"] = [f"Errore estrazione metriche: {e}"]

        return metrics

    def check_drift_alerts(self, metrics):
        """Controlla se ci sono alert di drift"""
        alerts = []

        # Alert per dataset drift
        if metrics["dataset_drift_detected"]:
            alerts.append(
                f"⚠️ Dataset drift rilevato! Score: {metrics['drift_score']:.2f}"
            )

        # Alert per drift score alto
        if metrics["drift_score"] > DRIFT_THRESHOLDS["statistical_test_threshold"]:
            alerts.append(f"⚠️ Drift score alto: {metrics['drift_score']:.2f}")

        # Alert per numero di colonne con drift
        drifted_count = len(metrics["drifted_columns"])
        total_features = len(FEATURE_CONFIG["numerical_features"])
        drift_percentage = drifted_count / total_features if total_features > 0 else 0

        if drift_percentage > 0.3:  # Più del 30% delle feature hanno drift
            alerts.append(
                f"⚠️ Troppe colonne con drift: {drifted_count}/{total_features} ({drift_percentage:.1%})"
            )

        # Alert per colonne specifiche con drift significativo
        for column_drift in metrics["drifted_columns"]:
            if column_drift["drift_score"] > 0.5:  # Drift molto significativo
                alerts.append(
                    f"⚠️ Drift significativo in {column_drift['column']}: {column_drift['drift_score']:.2f}"
                )

        # Salva alert
        if alerts:
            alert_path = (
                self.paths["alerts_dir"]
                / f"drift_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "data_drift",
                "alerts": alerts,
                "metrics": metrics,
            }
            with open(alert_path, "w") as f:
                json.dump(alert_data, f, indent=2)
            logger.warning(f"⚠️ Alert drift salvati: {alert_path}")

        return alerts

    def analyze_drift_causes(self, metrics):
        """Analizza le possibili cause del drift"""
        causes = []

        if metrics["drifted_columns"]:
            # Analizza le colonne con drift
            high_drift_columns = [
                col for col in metrics["drifted_columns"] if col["drift_score"] > 0.3
            ]

            if high_drift_columns:
                causes.append("Colonne con drift significativo:")
                for col in high_drift_columns:
                    causes.append(
                        f"  - {col['column']}: score={col['drift_score']:.2f}"
                    )

            # Analizza pattern di drift
            drift_scores = [col["drift_score"] for col in metrics["drifted_columns"]]
            if drift_scores:
                avg_drift = np.mean(drift_scores)
                causes.append(f"Drift medio nelle colonne affette: {avg_drift:.2f}")

        return causes


def main():
    """Funzione principale per il drift detection"""
    detector = DriftDetector()

    # Path dei dati
    reference_data_path = Path("data/processed/train_set.csv")
    current_data_path = Path("data/processed/test_set.csv")  # Per test

    # Carica dati
    reference_data = detector.load_reference_data(reference_data_path)
    current_data = detector.load_current_data(current_data_path)

    if reference_data is not None and current_data is not None:
        # Genera report
        metrics, report_path = detector.generate_drift_report(
            reference_data, current_data
        )

        if metrics:
            # Controlla alert
            alerts = detector.check_drift_alerts(metrics)

            # Analizza cause
            causes = detector.analyze_drift_causes(metrics)

            print(f"\n📊 Data Drift Report")
            print(
                f"Dataset drift rilevato: {metrics.get('dataset_drift_detected', False)}"
            )
            print(f"Drift score: {metrics.get('drift_score', 0):.2f}")
            summary = metrics.get("summary", {})
            print(
                f"Colonne con drift: {len(metrics.get('drifted_columns', []))}/{summary.get('total_features', 'N/A')}"
            )
            print(f"Percentuale drift: {summary.get('drift_percentage', 0):.1%}")

            if alerts:
                print(f"\n⚠️ Alert rilevati:")
                for alert in alerts:
                    print(f"  - {alert}")
            else:
                print(f"\n✅ Nessun alert di drift rilevato")

            if causes:
                print(f"\n🔍 Analisi cause:")
                for cause in causes:
                    print(f"  {cause}")

            print(f"\n📄 Report salvato: {report_path}")
        else:
            print("❌ Errore nella generazione del report")
    else:
        print("❌ Impossibile caricare i dati")


if __name__ == "__main__":
    main()
