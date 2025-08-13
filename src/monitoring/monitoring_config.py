"""
Configurazione centralizzata per il monitoring con Evidently AI
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MONITORING_DIR = PROJECT_ROOT / "monitoring"

# Configurazione dual-mode
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # local, cloud

if ENVIRONMENT == "cloud":
    # Configurazione per ambiente cloud (GCP)
    EVIDENTLY_PORT = 8080
    EVIDENTLY_HOST = "0.0.0.0"

    # GCS Buckets per cloud
    DATA_BUCKET = os.getenv("DATA_BUCKET", "mlops-breast-cancer-data")
    MONITORING_BUCKET = os.getenv("MONITORING_BUCKET", "mlops-breast-cancer-monitoring")
    MODELS_BUCKET = os.getenv("MODELS_BUCKET", "mlops-breast-cancer-models")

    # Evidently Dashboard URL
    EVIDENTLY_DASHBOARD_URL = (
        "https://evidently-dashboard-403815755558.europe-west1.run.app"
    )

    print("🌤️  Configurazione Monitoring: AMBIENTE CLOUD")
else:
    # Configurazione per ambiente locale
    EVIDENTLY_PORT = 8080
    EVIDENTLY_HOST = "0.0.0.0"

    # Path locali
    DATA_BUCKET = "local"
    MONITORING_BUCKET = "local"
    MODELS_BUCKET = "local"

    # Evidently Dashboard URL locale
    EVIDENTLY_DASHBOARD_URL = "http://localhost:8080"

    print("🏠  Configurazione Monitoring: AMBIENTE LOCALE")

# Data Quality Thresholds
DATA_QUALITY_THRESHOLDS = {
    "missing_values_threshold": 0.1,  # 10% massimo valori mancanti
    "duplicate_rows_threshold": 0.05,  # 5% massimo righe duplicate
    "outliers_threshold": 0.05,  # 5% massimo outliers
    "data_types_consistency": True,
    "range_validation": True,
}

# Drift Detection Thresholds
DRIFT_THRESHOLDS = {
    "statistical_test_threshold": 0.05,  # p-value per test statistici
    "psi_threshold": 0.25,  # Population Stability Index
    "kl_divergence_threshold": 0.1,  # KL Divergence
    "wasserstein_threshold": 0.1,  # Wasserstein Distance
}

# Performance Monitoring Thresholds
PERFORMANCE_THRESHOLDS = {
    "accuracy_threshold": 0.85,
    "recall_threshold": 0.90,  # Importante per medical diagnosis
    "precision_threshold": 0.80,
    "f1_threshold": 0.85,
    "prediction_drift_threshold": 0.1,
}

# Monitoring Schedule
MONITORING_SCHEDULE = {
    "data_quality_check": "daily",
    "drift_detection": "weekly",
    "performance_monitoring": "daily",
    "dashboard_refresh": "hourly",
}

# Alert Configuration
ALERT_CONFIG = {
    "email_alerts": False,  # Per ora disabilitato
    "slack_alerts": False,  # Per ora disabilitato
    "log_alerts": True,
    "dashboard_alerts": True,
}

# Feature Configuration per Evidently
FEATURE_CONFIG = {
    "numerical_features": [
        "radius_mean",
        "perimeter_mean",
        "area_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "radius_se",
        "radius_worst",
        "perimeter_worst",
        "area_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "mean_features_avg",
        "mean_features_std",
        "worst_features_avg",
        "worst_features_max",
        "radius_ratio",
        "perimeter_ratio",
        "area_ratio",
    ],
    "categorical_features": [],
    "target_column": "target",
}

# Evidently Report Types
REPORT_TYPES = {
    "data_quality": "data_quality",
    "data_drift": "data_drift",
    "target_drift": "target_drift",
    "model_performance": "classification_performance",
    "model_monitoring": "classification_performance",
}


def ensure_directories():
    """Crea le directory necessarie per il monitoring"""
    directories = [
        MONITORING_DIR,
        MONITORING_DIR / "reports",
        MONITORING_DIR / "dashboards",
        MONITORING_DIR / "alerts",
        MONITORING_DIR / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory creata: {directory}")


def get_monitoring_paths():
    """Restituisce i path per i file di monitoring"""
    return {
        "reports_dir": MONITORING_DIR / "reports",
        "dashboards_dir": MONITORING_DIR / "dashboards",
        "alerts_dir": MONITORING_DIR / "alerts",
        "logs_dir": MONITORING_DIR / "logs",
        "data_quality_report": MONITORING_DIR / "reports" / "data_quality_report.html",
        "drift_report": MONITORING_DIR / "reports" / "drift_report.html",
        "performance_report": MONITORING_DIR / "reports" / "performance_report.html",
        "dashboard": MONITORING_DIR / "dashboards" / "monitoring_dashboard.html",
    }


if __name__ == "__main__":
    ensure_directories()
    print("✓ Configurazione monitoring completata")
