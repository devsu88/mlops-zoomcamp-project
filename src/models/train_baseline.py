#!/usr/bin/env python3
"""
Script per training dei modelli baseline per classificazione breast cancer.
Include Logistic Regression, Random Forest, SVM e XGBoost.
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def load_processed_data():
    """
    Carica i dati processati (train e test sets).
    """
    logger.info("=== CARICAMENTO DATI PROCESSATI ===")

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
    logger.info(f"Features: {list(X_train.columns)}")
    logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def create_baseline_models():
    """
    Crea i modelli baseline con configurazioni ottimizzate.
    """
    logger.info("=== CREAZIONE MODELLI BASELINE ===")

    models = {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",  # Gestisce class imbalance
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,  # Necessario per ROC-AUC
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            scale_pos_weight=1.7,  # Gestisce imbalance (212/357 ≈ 1.7)
        ),
    }

    logger.info(f"Modelli creati: {list(models.keys())}")
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Valuta un singolo modello con cross-validation e test set.
    """
    logger.info(f"=== VALUTAZIONE {model_name.upper()} ===")

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
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità classe positiva

    # Metriche su test set
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # AUC scores
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Results
    results = {
        "model_name": model_name,
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
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

    logger.info(f"Test Results - {model_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")

    return results, model


def save_model(model, model_name, results_dir, results):
    """
    Salva il modello e i risultati.
    """
    # Creare directory se non esistente
    results_dir.mkdir(exist_ok=True)

    # Salvare modello
    model_path = results_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")

    # Salvare risultati
    results_path = results_dir / f"{model_name}_results.json"

    # Convertire numpy arrays per JSON serialization
    results_json = {
        "model_name": results["model_name"],
        "cv_recall_mean": float(results["cv_recall_mean"]),
        "cv_recall_std": float(results["cv_recall_std"]),
        "cv_f1_mean": float(results["cv_f1_mean"]),
        "cv_f1_std": float(results["cv_f1_std"]),
        "cv_accuracy_mean": float(results["cv_accuracy_mean"]),
        "cv_accuracy_std": float(results["cv_accuracy_std"]),
        "test_accuracy": float(results["test_accuracy"]),
        "test_precision": float(results["test_precision"]),
        "test_recall": float(results["test_recall"]),
        "test_f1": float(results["test_f1"]),
        "test_roc_auc": float(results["test_roc_auc"]),
        "test_pr_auc": float(results["test_pr_auc"]),
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    import json

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Risultati salvati: {results_path}")


def create_comparison_report(all_results, results_dir):
    """
    Crea un report di confronto tra tutti i modelli.
    """
    logger.info("=== CREAZIONE REPORT CONFRONTO ===")

    # DataFrame per confronto
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append(
            {
                "Model": model_name,
                "CV_Recall_Mean": results["cv_recall_mean"],
                "CV_F1_Mean": results["cv_f1_mean"],
                "CV_Accuracy_Mean": results["cv_accuracy_mean"],
                "Test_Recall": results["test_recall"],
                "Test_F1": results["test_f1"],
                "Test_Precision": results["test_precision"],
                "Test_Accuracy": results["test_accuracy"],
                "Test_ROC_AUC": results["test_roc_auc"],
                "Test_PR_AUC": results["test_pr_auc"],
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Salvare report
    report_path = results_dir / "baseline_models_comparison.csv"
    comparison_df.to_csv(report_path, index=False)
    logger.info(f"Report confronto salvato: {report_path}")

    # Log del confronto
    logger.info("\n=== CONFRONTO MODELLI BASELINE ===")
    logger.info(comparison_df.to_string(index=False))

    return comparison_df


def plot_results(all_results, results_dir):
    """
    Crea grafici per confrontare i modelli.
    """
    logger.info("=== CREAZIONE GRAFICI ===")

    # Creare directory per grafici
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Confronto metriche principali
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    models = list(all_results.keys())
    metrics = ["test_recall", "test_f1", "test_pr_auc", "test_roc_auc"]
    metric_names = ["Recall", "F1-Score", "PR-AUC", "ROC-AUC"]

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]
        values = [all_results[model][metric] for model in models]

        bars = ax.bar(
            models, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        )
        ax.set_title(f"{name} Comparison")
        ax.set_ylabel(name)
        ax.set_ylim(0, 1)

        # Aggiungere valori sulle barre
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(
        plots_dir / "baseline_models_comparison.png", dpi=300, bbox_inches="tight"
    )
    logger.info(
        f"Grafico confronto salvato: {plots_dir / 'baseline_models_comparison.png'}"
    )

    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, (model_name, results) in enumerate(all_results.items()):
        cm = results["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{model_name.upper()} - Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    logger.info(f"Confusion matrices salvate: {plots_dir / 'confusion_matrices.png'}")

    plt.close("all")


def main():
    """
    Funzione principale per training modelli baseline.
    """
    logger.info("=== INIZIO TRAINING MODELLI BASELINE ===")

    try:
        # 1. Caricamento dati
        X_train, X_test, y_train, y_test = load_processed_data()

        # 2. Creazione modelli
        models = create_baseline_models()

        # 3. Training e valutazione
        all_results = {}
        trained_models = {}

        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            results, trained_model = evaluate_model(
                model, X_train, X_test, y_train, y_test, model_name
            )
            all_results[model_name] = results
            trained_models[model_name] = trained_model

            # Salvare modello e risultati
            save_model(trained_model, model_name, RESULTS_DIR, results)

        # 4. Report di confronto
        comparison_df = create_comparison_report(all_results, RESULTS_DIR)

        # 5. Grafici
        plot_results(all_results, RESULTS_DIR)

        # 6. Selezione best model
        best_model_name = comparison_df.loc[
            comparison_df["Test_Recall"].idxmax(), "Model"
        ]
        best_recall = comparison_df.loc[
            comparison_df["Test_Recall"].idxmax(), "Test_Recall"
        ]

        logger.info(f"\n{'='*50}")
        logger.info(f"🏆 BEST MODEL: {best_model_name}")
        logger.info(f"🏆 BEST RECALL: {best_recall:.4f}")
        logger.info(f"{'='*50}")

        logger.info("=== TRAINING MODELLI BASELINE COMPLETATO ===")

        return {
            "best_model": best_model_name,
            "best_recall": best_recall,
            "comparison_df": comparison_df,
            "all_results": all_results,
        }

    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        raise


if __name__ == "__main__":
    main()
