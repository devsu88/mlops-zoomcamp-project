#!/usr/bin/env python3
"""
Script per training dei modelli con integrazione MLflow.
Include experiment tracking, model registry e logging completo.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from .mlflow_config import create_mlflow_run, setup_mlflow
from api.config import is_cloud_environment, get_api_config

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione percorsi
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"
RANDOM_STATE = 42
CV_FOLDS = 5


def load_processed_data():
    """
    Carica i dati processati (train e test sets).
    """
    logger.info("=== CARICAMENTO DATI PROCESSATI ===")

    if is_cloud_environment():
        # Per cloud, caricare da GCS
        api_config = get_api_config()
        from google.cloud import storage
        import tempfile

        logger.info("🌤️  Caricamento dati da Cloud Storage")

        # Download temporaneo dei dati
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as tmp_train, tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as tmp_test:
            storage_client = storage.Client()
            bucket = storage_client.bucket(api_config["data_bucket"])

            # Download train set
            blob = bucket.blob("processed/train_set.csv")
            blob.download_to_filename(tmp_train.name)
            train_df = pd.read_csv(tmp_train.name)

            # Download test set
            blob = bucket.blob("processed/test_set.csv")
            blob.download_to_filename(tmp_test.name)
            test_df = pd.read_csv(tmp_test.name)

    else:
        # Per locale, caricare da file system
        logger.info("🏠  Caricamento dati da File System Locale")
        train_path = PROCESSED_DATA_DIR / "train_set.csv"
        test_path = PROCESSED_DATA_DIR / "test_set.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    # Separare features e target
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def create_models_with_params():
    """
    Crea i modelli con parametri per MLflow tracking.
    """
    logger.info("=== CREAZIONE MODELLI CON PARAMETRI ===")

    models_config = {
        "logistic_regression": {
            "model": LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"
            ),
            "params": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "svm": {
            "model": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "xgboost": {
            "model": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                scale_pos_weight=1.7,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1.7,
                "random_state": RANDOM_STATE,
            },
        },
    }

    logger.info(f"Modelli creati: {list(models_config.keys())}")
    return models_config


def evaluate_model_with_mlflow(
    model_config, X_train, X_test, y_train, y_test, model_name
):
    """
    Valuta un modello e logga tutto in MLflow.
    """
    logger.info(f"=== VALUTAZIONE {model_name.upper()} CON MLFLOW ===")

    model = model_config["model"]
    params = model_config["params"]

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Metriche per CV
    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    logger.info(f"CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    logger.info(f"CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    logger.info(
        f"CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})"
    )

    # Training su tutto il train set
    model.fit(X_train, y_train)

    # Predizioni
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metriche su test set
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Preparare metriche per MLflow
    metrics = {
        "cv_recall_mean": cv_recall.mean(),
        "cv_recall_std": cv_recall.std(),
        "cv_f1_mean": cv_f1.mean(),
        "cv_f1_std": cv_f1.std(),
        "cv_accuracy_mean": cv_accuracy.mean(),
        "cv_accuracy_std": cv_accuracy.std(),
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
    }

    # Logga in MLflow
    artifacts_path = RESULTS_DIR / "plots" if (RESULTS_DIR / "plots").exists() else None
    model_details = create_mlflow_run(
        model_name, params, metrics, artifacts_path, model
    )

    logger.info(f"Test Results - {model_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")

    return {
        "model": model,
        "metrics": metrics,
        "model_details": model_details,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "confusion_matrix": cm,
    }


def save_model_locally(model, model_name, results_dir):
    """
    Salva il modello localmente come backup.
    """
    results_dir.mkdir(exist_ok=True)
    model_path = results_dir / f"{model_name}_mlflow.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato localmente: {model_path}")


def create_comparison_report_mlflow(all_results, results_dir):
    """
    Crea un report di confronto per i modelli MLflow.
    """
    logger.info("=== CREAZIONE REPORT CONFRONTO MLFLOW ===")

    comparison_data = []
    for model_name, results in all_results.items():
        metrics = results["metrics"]
        comparison_data.append(
            {
                "Model": model_name,
                "CV_Recall_Mean": metrics["cv_recall_mean"],
                "CV_F1_Mean": metrics["cv_f1_mean"],
                "CV_Accuracy_Mean": metrics["cv_accuracy_mean"],
                "Test_Recall": metrics["test_recall"],
                "Test_F1": metrics["test_f1"],
                "Test_Precision": metrics["test_precision"],
                "Test_Accuracy": metrics["test_accuracy"],
                "Test_ROC_AUC": metrics["test_roc_auc"],
                "Test_PR_AUC": metrics["test_pr_auc"],
                "MLflow_Run_ID": results["model_details"].run_id
                if results["model_details"]
                else "N/A",
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Salvare report
    report_path = results_dir / "mlflow_models_comparison.csv"
    comparison_df.to_csv(report_path, index=False)
    logger.info(f"Report MLflow salvato: {report_path}")

    # Log del confronto
    logger.info("\n=== CONFRONTO MODELLI MLFLOW ===")
    logger.info(comparison_df.to_string(index=False))

    return comparison_df


def main():
    """
    Funzione principale per training con MLflow.
    """
    logger.info("=== INIZIO TRAINING CON MLFLOW ===")

    try:
        # 1. Setup MLflow
        experiment_id = setup_mlflow()

        # 2. Caricamento dati
        X_train, X_test, y_train, y_test = load_processed_data()

        # 3. Creazione modelli
        models_config = create_models_with_params()

        # 4. Training e valutazione con MLflow
        all_results = {}

        for model_name, model_config in models_config.items():
            logger.info(f"\n{'='*50}")
            results = evaluate_model_with_mlflow(
                model_config, X_train, X_test, y_train, y_test, model_name
            )
            all_results[model_name] = results

            # Salvare modello localmente come backup
            save_model_locally(results["model"], model_name, RESULTS_DIR)

        # 5. Report di confronto
        comparison_df = create_comparison_report_mlflow(all_results, RESULTS_DIR)

        # 6. Selezione best model
        best_model_name = comparison_df.loc[
            comparison_df["Test_Recall"].idxmax(), "Model"
        ]
        best_recall = comparison_df.loc[
            comparison_df["Test_Recall"].idxmax(), "Test_Recall"
        ]

        logger.info(f"\n{'='*50}")
        logger.info(f"🏆 BEST MODEL (MLflow): {best_model_name}")
        logger.info(f"🏆 BEST RECALL: {best_recall:.4f}")
        logger.info(f"{'='*50}")

        logger.info("=== TRAINING CON MLFLOW COMPLETATO ===")

        return {
            "best_model": best_model_name,
            "best_recall": best_recall,
            "comparison_df": comparison_df,
            "all_results": all_results,
            "experiment_id": experiment_id,
        }

    except Exception as e:
        logger.error(f"Errore durante il training con MLflow: {e}")
        raise


if __name__ == "__main__":
    main()
