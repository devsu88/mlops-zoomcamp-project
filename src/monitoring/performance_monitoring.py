"""
Model Performance Monitoring con Evidently AI
Monitora le performance del modello in produzione
"""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    ClassificationQualityMetric,
)
from evidently.report import Report

from src.monitoring.monitoring_config import (
    FEATURE_CONFIG,
    PERFORMANCE_THRESHOLDS,
    ensure_directories,
    get_monitoring_paths,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor per le performance del modello"""

    def __init__(self):
        self.paths = get_monitoring_paths()
        ensure_directories()
        self.column_mapping = ColumnMapping(
            target=FEATURE_CONFIG["target_column"],
            numerical_features=FEATURE_CONFIG["numerical_features"],
            categorical_features=FEATURE_CONFIG["categorical_features"],
        )

    def load_model(self, model_path):
        """Carica il modello addestrato"""
        try:
            import joblib

            model = joblib.load(model_path)
            logger.info(f"✓ Modello caricato: {model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Errore nel caricamento modello: {e}")
            return None

    def load_scaler(self, scaler_path):
        """Carica lo scaler"""
        try:
            import joblib

            scaler = joblib.load(scaler_path)
            logger.info(f"✓ Scaler caricato: {scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"❌ Errore nel caricamento scaler: {e}")
            return None

    def load_reference_data(self, data_path):
        """Carica i dati di riferimento (training set con predizioni)"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✓ Dati di riferimento caricati: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Errore nel caricamento dati di riferimento: {e}")
            return None

    def load_current_data(self, data_path):
        """Carica i dati attuali (production data con predizioni)"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✓ Dati attuali caricati: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Errore nel caricamento dati attuali: {e}")
            return None

    def generate_predictions(self, model, scaler, data, target_column=None):
        """Genera predizioni per i dati"""
        try:
            # Seleziona solo le feature numeriche
            feature_columns = [
                col
                for col in FEATURE_CONFIG["numerical_features"]
                if col in data.columns
            ]
            X = data[feature_columns]

            # Applica scaling
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X

            # Genera predizioni
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)

            # Aggiungi predizioni al dataframe
            data_with_predictions = data.copy()
            data_with_predictions["prediction"] = predictions
            data_with_predictions["prediction_probability"] = probabilities[
                :, 1
            ]  # Probabilità classe positiva

            logger.info(f"✓ Predizioni generate per {len(data)} campioni")
            return data_with_predictions

        except Exception as e:
            logger.error(f"❌ Errore nella generazione predizioni: {e}")
            return None

    def generate_performance_report(self, reference_data, current_data):
        """Genera report di performance con Evidently"""
        try:
            # Genera report performance generale
            performance_report = Report(
                metrics=[
                    ClassificationPreset(),
                    ClassificationQualityMetric(),
                    ClassificationConfusionMatrix(),
                    ClassificationQualityByClass(),
                ]
            )

            performance_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Salva report HTML
            report_path = self.paths["performance_report"]
            performance_report.save_html(report_path)
            logger.info(f"✓ Report performance salvato: {report_path}")

            # Estrai metriche chiave
            metrics = self._extract_performance_metrics(performance_report)

            # Salva metriche JSON
            metrics_path = self.paths["reports_dir"] / "performance_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"✓ Metriche performance salvate: {metrics_path}")

            return metrics, report_path

        except Exception as e:
            logger.error(f"❌ Errore nella generazione report performance: {e}")
            return None, None

    def _extract_performance_metrics(self, performance_report):
        """Estrae metriche chiave dal report Evidently"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "reference_performance": {},
            "current_performance": {},
            "performance_drift": {},
            "summary": {},
        }

        try:
            # Estrai metriche dal report
            report_data = performance_report.json()

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

                    if metric_name == "ClassificationQualityMetric":
                        # Estrai informazioni sulla qualità della classificazione
                        current_data = metric_result.get("current", {})
                        reference_data = metric_result.get("reference", {})

                        logger.info(
                            f"🔍 ClassificationQualityMetric - current_data keys: {list(current_data.keys()) if isinstance(current_data, dict) else 'not dict'}"
                        )

                        if isinstance(current_data, dict):
                            metrics["current_performance"] = {
                                "accuracy": current_data.get("accuracy", 0.0),
                                "precision": current_data.get("precision", 0.0),
                                "recall": current_data.get("recall", 0.0),
                                "f1": current_data.get("f1", 0.0),
                            }

                        if isinstance(reference_data, dict):
                            metrics["reference_performance"] = {
                                "accuracy": reference_data.get("accuracy", 0.0),
                                "precision": reference_data.get("precision", 0.0),
                                "recall": reference_data.get("recall", 0.0),
                                "f1": reference_data.get("f1", 0.0),
                            }

                    elif metric_name == "ClassificationConfusionMatrix":
                        # Estrai informazioni sulla confusion matrix
                        current_data = metric_result.get("current", {})
                        reference_data = metric_result.get("reference", {})

                        logger.info(
                            f"🔍 ClassificationConfusionMatrix - current_data keys: {list(current_data.keys()) if isinstance(current_data, dict) else 'not dict'}"
                        )

                        if isinstance(current_data, dict) or isinstance(
                            reference_data, dict
                        ):
                            metrics["confusion_matrix"] = {
                                "reference": reference_data
                                if isinstance(reference_data, dict)
                                else {},
                                "current": current_data
                                if isinstance(current_data, dict)
                                else {},
                            }

                    elif metric_name == "ClassificationQualityByClass":
                        # Estrai informazioni per classe
                        current_data = metric_result.get("current", {})
                        reference_data = metric_result.get("reference", {})

                        logger.info(
                            f"🔍 ClassificationQualityByClass - current_data keys: {list(current_data.keys()) if isinstance(current_data, dict) else 'not dict'}"
                        )

                        if isinstance(current_data, dict) or isinstance(
                            reference_data, dict
                        ):
                            metrics["class_performance"] = {
                                "reference": reference_data
                                if isinstance(reference_data, dict)
                                else {},
                                "current": current_data
                                if isinstance(current_data, dict)
                                else {},
                            }

            # Calcola drift delle performance
            if metrics.get("reference_performance") and metrics.get(
                "current_performance"
            ):
                drift_metrics = {}
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    ref_val = metrics["reference_performance"].get(metric, 0.0)
                    curr_val = metrics["current_performance"].get(metric, 0.0)
                    if ref_val > 0:
                        drift = (curr_val - ref_val) / ref_val
                        drift_metrics[f"{metric}_drift"] = drift

                metrics["performance_drift"] = drift_metrics

            # Aggiungi summary
            current_data = report_data.get("current", {})
            if isinstance(current_data, dict):
                data_info = current_data.get("data", {})
                if isinstance(data_info, dict):
                    target_data = data_info.get("target", [])
                    if isinstance(target_data, list):
                        metrics["summary"] = {
                            "total_samples": len(target_data),
                            "positive_class_samples": sum(
                                1 for x in target_data if x == 1
                            ),
                            "negative_class_samples": sum(
                                1 for x in target_data if x == 0
                            ),
                        }
                    else:
                        metrics["summary"] = {
                            "total_samples": 0,
                            "positive_class_samples": 0,
                            "negative_class_samples": 0,
                        }
                else:
                    metrics["summary"] = {
                        "total_samples": 0,
                        "positive_class_samples": 0,
                        "negative_class_samples": 0,
                    }
            else:
                metrics["summary"] = {
                    "total_samples": 0,
                    "positive_class_samples": 0,
                    "negative_class_samples": 0,
                }

        except Exception as e:
            logger.error(f"❌ Errore nell'estrazione metriche performance: {e}")
            metrics["issues"] = [f"Errore estrazione metriche: {e}"]

        return metrics

    def check_performance_alerts(self, metrics):
        """Controlla se ci sono alert di performance"""
        alerts = []

        # Alert per performance attuali
        current_perf = metrics.get("current_performance", {})

        if (
            current_perf.get("accuracy", 1.0)
            < PERFORMANCE_THRESHOLDS["accuracy_threshold"]
        ):
            alerts.append(f"⚠️ Accuracy bassa: {current_perf['accuracy']:.2f}")

        if current_perf.get("recall", 1.0) < PERFORMANCE_THRESHOLDS["recall_threshold"]:
            alerts.append(f"⚠️ Recall bassa: {current_perf['recall']:.2f}")

        if (
            current_perf.get("precision", 1.0)
            < PERFORMANCE_THRESHOLDS["precision_threshold"]
        ):
            alerts.append(f"⚠️ Precision bassa: {current_perf['precision']:.2f}")

        if current_perf.get("f1", 1.0) < PERFORMANCE_THRESHOLDS["f1_threshold"]:
            alerts.append(f"⚠️ F1-score basso: {current_perf['f1']:.2f}")

        # Alert per drift delle performance
        drift_metrics = metrics.get("performance_drift", {})

        for metric, drift in drift_metrics.items():
            if abs(drift) > PERFORMANCE_THRESHOLDS["prediction_drift_threshold"]:
                direction = "peggioramento" if drift < 0 else "miglioramento"
                alerts.append(f"⚠️ Drift {metric}: {drift:.2%} ({direction})")

        # Salva alert
        if alerts:
            alert_path = (
                self.paths["alerts_dir"]
                / f"performance_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "model_performance",
                "alerts": alerts,
                "metrics": metrics,
            }
            with open(alert_path, "w") as f:
                json.dump(alert_data, f, indent=2)
            logger.warning(f"⚠️ Alert performance salvati: {alert_path}")

        return alerts

    def analyze_performance_trends(self, metrics):
        """Analizza i trend delle performance"""
        trends = []

        current_perf = metrics.get("current_performance", {})
        reference_perf = metrics.get("reference_performance", {})
        drift_metrics = metrics.get("performance_drift", {})

        if current_perf and reference_perf:
            # Analizza metriche principali
            for metric in ["accuracy", "precision", "recall", "f1"]:
                curr_val = current_perf.get(metric, 0.0)
                ref_val = reference_perf.get(metric, 0.0)
                drift = drift_metrics.get(f"{metric}_drift", 0.0)

                if abs(drift) > 0.05:  # Drift significativo
                    trend = "miglioramento" if drift > 0 else "peggioramento"
                    trends.append(f"{metric.capitalize()}: {trend} ({drift:.1%})")

        return trends


def main():
    """Funzione principale per il performance monitoring"""
    monitor = PerformanceMonitor()

    # Path dei file
    model_path = Path(
        "mlruns/345803254500172789/c0edfdfcb43c46758634223da7d1faa5/artifacts/logistic_regression/model.pkl"
    )
    scaler_path = Path("data/processed/scaler.joblib")
    reference_data_path = Path("data/processed/train_set.csv")
    current_data_path = Path("data/processed/test_set.csv")

    # Carica modello e scaler
    model = monitor.load_model(model_path)
    scaler = monitor.load_scaler(scaler_path)

    if model is not None and scaler is not None:
        # Carica dati
        reference_data = monitor.load_reference_data(reference_data_path)
        current_data = monitor.load_current_data(current_data_path)

        if reference_data is not None and current_data is not None:
            # Genera predizioni
            reference_with_predictions = monitor.generate_predictions(
                model, scaler, reference_data
            )
            current_with_predictions = monitor.generate_predictions(
                model, scaler, current_data
            )

            if (
                reference_with_predictions is not None
                and current_with_predictions is not None
            ):
                # Genera report
                metrics, report_path = monitor.generate_performance_report(
                    reference_with_predictions, current_with_predictions
                )

                if metrics:
                    # Controlla alert
                    alerts = monitor.check_performance_alerts(metrics)

                    # Analizza trend
                    trends = monitor.analyze_performance_trends(metrics)

                    print(f"\n📊 Model Performance Report")

                    if metrics.get("current_performance"):
                        curr = metrics["current_performance"]
                        print(f"Accuracy: {curr.get('accuracy', 0):.3f}")
                        print(f"Precision: {curr.get('precision', 0):.3f}")
                        print(f"Recall: {curr.get('recall', 0):.3f}")
                        print(f"F1-Score: {curr.get('f1', 0):.3f}")

                    if metrics.get("performance_drift"):
                        print(f"\n📈 Performance Drift:")
                        for metric, drift in metrics["performance_drift"].items():
                            print(f"  {metric}: {drift:.1%}")

                    if alerts:
                        print(f"\n⚠️ Alert rilevati:")
                        for alert in alerts:
                            print(f"  - {alert}")
                    else:
                        print(f"\n✅ Nessun alert di performance rilevato")

                    if trends:
                        print(f"\n📈 Trend performance:")
                        for trend in trends:
                            print(f"  - {trend}")

                    print(f"\n📄 Report salvato: {report_path}")
                else:
                    print("❌ Errore nella generazione del report")
            else:
                print("❌ Errore nella generazione delle predizioni")
        else:
            print("❌ Impossibile caricare i dati")
    else:
        print("❌ Impossibile caricare modello o scaler")


if __name__ == "__main__":
    main()
