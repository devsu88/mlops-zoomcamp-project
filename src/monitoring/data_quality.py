"""
Data Quality Monitoring con Evidently AI
Controlla la qualità dei dati in ingresso e genera report
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataQualityPreset
from evidently.metrics import DataQualityStabilityMetric
from evidently.report import Report

from src.monitoring.monitoring_config import (DATA_QUALITY_THRESHOLDS,
                                              FEATURE_CONFIG,
                                              ensure_directories,
                                              get_monitoring_paths)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor per la qualità dei dati"""

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

    def validate_data_structure(self, df):
        """Valida la struttura dei dati"""
        issues = []

        # Controlla colonne richieste
        required_columns = FEATURE_CONFIG["numerical_features"] + [
            FEATURE_CONFIG["target_column"]
        ]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Colonne mancanti: {missing_columns}")

        # Controlla tipi di dati
        for col in FEATURE_CONFIG["numerical_features"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Colonna {col} non è numerica")

        # Controlla valori mancanti
        missing_pct = df[FEATURE_CONFIG["numerical_features"]].isnull().sum() / len(df)
        high_missing = missing_pct[
            missing_pct > DATA_QUALITY_THRESHOLDS["missing_values_threshold"]
        ]
        if not high_missing.empty:
            issues.append(f"Troppi valori mancanti: {high_missing.to_dict()}")

        return issues

    def generate_data_quality_report(self, reference_data, current_data=None):
        """Genera report di qualità dati con Evidently"""
        try:
            # Se non ci sono dati attuali, usa i dati di riferimento come baseline
            if current_data is None:
                current_data = reference_data.copy()

            # Valida struttura dati
            ref_issues = self.validate_data_structure(reference_data)
            curr_issues = self.validate_data_structure(current_data)

            if ref_issues:
                logger.warning(f"⚠️ Problemi nei dati di riferimento: {ref_issues}")
            if curr_issues:
                logger.warning(f"⚠️ Problemi nei dati attuali: {curr_issues}")

            # Genera report Evidently
            data_quality_report = Report(
                metrics=[DataQualityPreset(), DataQualityStabilityMetric()]
            )

            data_quality_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Salva report HTML
            report_path = self.paths["data_quality_report"]
            data_quality_report.save_html(report_path)
            logger.info(f"✓ Report qualità dati salvato: {report_path}")

            # Estrai metriche chiave
            metrics = self._extract_quality_metrics(data_quality_report)

            # Salva metriche JSON
            metrics_path = self.paths["reports_dir"] / "data_quality_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"✓ Metriche qualità dati salvate: {metrics_path}")

            return metrics, report_path

        except Exception as e:
            logger.error(f"❌ Errore nella generazione report qualità dati: {e}")
            return None, None

    def _extract_quality_metrics(self, report):
        """Estrae metriche chiave dal report Evidently"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "data_quality_score": 0.0,
            "issues": [],
            "summary": {},
        }

        try:
            # Estrai metriche dal report
            report_data = report.json()

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

            # Calcola score complessivo
            quality_metrics = []
            missing_pct = 0
            duplicate_pct = 0

            # Estrai metriche dall'array metrics
            metrics_array = report_data.get("metrics", [])
            logger.info(f"🔍 Numero di metriche trovate: {len(metrics_array)}")

            for metric_item in metrics_array:
                if isinstance(metric_item, dict):
                    metric_name = metric_item.get("metric", "")
                    metric_result = metric_item.get("result", {})

                    logger.info(f"🔍 Processando metrica: {metric_name}")

                    if metric_name == "DatasetSummaryMetric":
                        # Estrai informazioni dal dataset summary
                        current_data = metric_result.get("current", {})
                        logger.info(
                            f"🔍 DatasetSummaryMetric - current_data keys: {list(current_data.keys()) if isinstance(current_data, dict) else 'not dict'}"
                        )

                        if isinstance(current_data, dict):
                            # Conta righe totali
                            total_rows = current_data.get("number_of_rows", 0)
                            metrics["summary"]["total_rows"] = total_rows
                            logger.info(f"🔍 Righe totali trovate: {total_rows}")

                            # Conta colonne
                            columns_count = current_data.get("number_of_columns", 0)
                            metrics["summary"]["total_columns"] = columns_count
                            logger.info(f"🔍 Colonne totali trovate: {columns_count}")

                    elif metric_name == "DatasetMissingValuesMetric":
                        # Estrai informazioni sui valori mancanti
                        current_data = metric_result.get("current", {})
                        logger.info(
                            f"🔍 DatasetMissingValuesMetric - current_data keys: {list(current_data.keys()) if isinstance(current_data, dict) else 'not dict'}"
                        )

                        if isinstance(current_data, dict):
                            # Usa le chiavi corrette per i valori mancanti
                            total_missing = current_data.get(
                                "number_of_missing_values", 0
                            )
                            total_rows = current_data.get("number_of_rows", 0)
                            total_columns = current_data.get("number_of_columns", 0)
                            total_cells = total_rows * total_columns

                            logger.info(
                                f"🔍 Total missing: {total_missing}, total cells: {total_cells}"
                            )

                            if total_cells > 0:
                                missing_pct = total_missing / total_cells
                                quality_metrics.append(1 - missing_pct)
                                logger.info(f"🔍 Missing percentage: {missing_pct:.2%}")

                    elif metric_name == "DatasetDuplicatesMetric":
                        # Estrai informazioni sui duplicati
                        current_data = metric_result.get("current", {})
                        if isinstance(current_data, dict):
                            duplicate_pct = current_data.get("duplicates_share", 0)
                            quality_metrics.append(1 - duplicate_pct)

            # Calcola score finale
            if quality_metrics:
                metrics["data_quality_score"] = np.mean(quality_metrics)
            else:
                # Se non abbiamo metriche specifiche, usa un score basato sui dati disponibili
                if metrics["summary"]["total_rows"] > 0:
                    metrics[
                        "data_quality_score"
                    ] = 0.8  # Score di default per dati validi

            # Aggiungi summary finale
            metrics["summary"].update(
                {"missing_values_pct": missing_pct, "duplicate_rows_pct": duplicate_pct}
            )

        except Exception as e:
            logger.error(f"❌ Errore nell'estrazione metriche: {e}")
            metrics["issues"].append(f"Errore estrazione metriche: {e}")

        return metrics

    def check_data_quality_alerts(self, metrics):
        """Controlla se ci sono alert di qualità dati"""
        alerts = []

        if metrics.get("data_quality_score", 1.0) < 0.8:
            alerts.append(
                f"⚠️ Score qualità dati basso: {metrics.get('data_quality_score', 0):.2f}"
            )

        summary = metrics.get("summary", {})
        if (
            summary.get("missing_values_pct", 0)
            > DATA_QUALITY_THRESHOLDS["missing_values_threshold"]
        ):
            alerts.append(
                f"⚠️ Troppi valori mancanti: {summary.get('missing_values_pct', 0):.2%}"
            )

        if (
            summary.get("duplicate_rows_pct", 0)
            > DATA_QUALITY_THRESHOLDS["duplicate_rows_threshold"]
        ):
            alerts.append(
                f"⚠️ Troppe righe duplicate: {summary.get('duplicate_rows_pct', 0):.2%}"
            )

        # Salva alert
        if alerts:
            alert_path = (
                self.paths["alerts_dir"]
                / f"data_quality_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "data_quality",
                "alerts": alerts,
                "metrics": metrics,
            }
            with open(alert_path, "w") as f:
                json.dump(alert_data, f, indent=2)
            logger.warning(f"⚠️ Alert qualità dati salvati: {alert_path}")

        return alerts


def main():
    """Funzione principale per il data quality monitoring"""
    monitor = DataQualityMonitor()

    # Path dei dati
    reference_data_path = Path("data/processed/train_set.csv")
    current_data_path = Path("data/processed/test_set.csv")  # Per test

    # Carica dati
    reference_data = monitor.load_reference_data(reference_data_path)
    current_data = monitor.load_current_data(current_data_path)

    if reference_data is not None:
        # Genera report
        metrics, report_path = monitor.generate_data_quality_report(
            reference_data, current_data
        )

        if metrics:
            # Controlla alert
            alerts = monitor.check_data_quality_alerts(metrics)

            print(f"\n📊 Data Quality Report")
            print(f"Score qualità: {metrics.get('data_quality_score', 0):.2f}")
            summary = metrics.get("summary", {})
            print(f"Righe totali: {summary.get('total_rows', 'N/A')}")
            print(f"Valori mancanti: {summary.get('missing_values_pct', 0):.2%}")
            print(f"Righe duplicate: {summary.get('duplicate_rows_pct', 0):.2%}")

            if alerts:
                print(f"\n⚠️ Alert rilevati:")
                for alert in alerts:
                    print(f"  - {alert}")
            else:
                print(f"\n✅ Nessun alert rilevato")

            print(f"\n📄 Report salvato: {report_path}")
        else:
            print("❌ Errore nella generazione del report")
    else:
        print("❌ Impossibile caricare i dati di riferimento")


if __name__ == "__main__":
    main()
