#!/usr/bin/env python3
"""
Script per analizzare l'importanza delle features del modello ottimizzato.
Include analisi dei coefficienti, permutation importance e visualizzazioni.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .mlflow_config import create_mlflow_run, setup_mlflow

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione percorsi
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RANDOM_STATE = 42


def load_data_and_model():
    """
    Carica i dati processati e il modello ottimizzato.
    """
    logger.info("=== CARICAMENTO DATI E MODELLO ===")

    # Caricare dati
    train_path = PROCESSED_DATA_DIR / "train_set.csv"
    test_path = PROCESSED_DATA_DIR / "test_set.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Caricare modello ottimizzato
    model_path = RESULTS_DIR / "logistic_regression_gridsearch_tuned.joblib"
    model = joblib.load(model_path)

    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Modello caricato: {model_path}")

    return X_train, X_test, y_train, y_test, model


def analyze_coefficient_importance(model, feature_names):
    """
    Analizza l'importanza delle features basata sui coefficienti del modello.
    """
    logger.info("=== ANALISI IMPORTANZA COEFFICIENTI ===")

    # Coefficienti del modello
    coefficients = model.coef_[0]

    # Creare DataFrame con coefficienti
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )

    # Ordinare per valore assoluto dei coefficienti
    feature_importance_df = feature_importance_df.sort_values(
        "abs_coefficient", ascending=False
    )

    logger.info("Top 10 features per coefficiente:")
    for i, row in feature_importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

    return feature_importance_df


def analyze_permutation_importance(model, X_test, y_test, feature_names):
    """
    Analizza l'importanza delle features usando permutation importance.
    """
    logger.info("=== ANALISI PERMUTATION IMPORTANCE ===")

    # Calcolare permutation importance
    perm_importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="recall",  # Priorità massima per il nostro caso
    )

    # Creare DataFrame
    perm_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance": perm_importance.importances_mean,
            "permutation_importance_std": perm_importance.importances_std,
        }
    )

    # Ordinare per importanza
    perm_importance_df = perm_importance_df.sort_values(
        "permutation_importance", ascending=False
    )

    logger.info("Top 10 features per permutation importance:")
    for i, row in perm_importance_df.head(10).iterrows():
        logger.info(
            f"  {row['feature']}: {row['permutation_importance']:.4f} ± {row['permutation_importance_std']:.4f}"
        )

    return perm_importance_df


def create_feature_importance_plots(coef_df, perm_df, plots_dir):
    """
    Crea visualizzazioni per l'importanza delle features.
    """
    logger.info("=== CREAZIONE PLOTS IMPORTANZA FEATURES ===")

    plots_dir.mkdir(exist_ok=True)

    # 1. Coefficient importance plot
    plt.figure(figsize=(12, 8))
    top_features = coef_df.head(15)

    colors = ["red" if x < 0 else "blue" for x in top_features["coefficient"]]
    plt.barh(range(len(top_features)), top_features["coefficient"], color=colors)
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Coefficient Value")
    plt.title("Top 15 Features by Logistic Regression Coefficients")
    plt.grid(axis="x", alpha=0.3)

    # Aggiungere linee per zero
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_dir / "coefficient_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Permutation importance plot
    plt.figure(figsize=(12, 8))
    top_perm_features = perm_df.head(15)

    plt.barh(range(len(top_perm_features)), top_perm_features["permutation_importance"])
    plt.yticks(range(len(top_perm_features)), top_perm_features["feature"])
    plt.xlabel("Permutation Importance (Recall)")
    plt.title("Top 15 Features by Permutation Importance")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Comparison plot
    plt.figure(figsize=(14, 10))

    # Normalizzare i valori per confronto
    coef_normalized = coef_df["abs_coefficient"] / coef_df["abs_coefficient"].max()
    perm_normalized = (
        perm_df["permutation_importance"] / perm_df["permutation_importance"].max()
    )

    # Creare DataFrame per confronto
    comparison_df = pd.DataFrame(
        {
            "feature": coef_df["feature"],
            "coefficient_importance": coef_normalized,
            "permutation_importance": perm_normalized,
        }
    )

    # Top 10 features per confronto
    top_comparison = comparison_df.head(10)

    x = np.arange(len(top_comparison))
    width = 0.35

    plt.bar(
        x - width / 2,
        top_comparison["coefficient_importance"],
        width,
        label="Coefficient Importance",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        top_comparison["permutation_importance"],
        width,
        label="Permutation Importance",
        alpha=0.8,
    )

    plt.xlabel("Features")
    plt.ylabel("Normalized Importance")
    plt.title("Comparison: Coefficient vs Permutation Importance")
    plt.xticks(x, top_comparison["feature"], rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "importance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots salvati in: {plots_dir}")

    return comparison_df


def analyze_feature_correlations(X_train, feature_names):
    """
    Analizza le correlazioni tra le features più importanti.
    """
    logger.info("=== ANALISI CORRELAZIONI FEATURES ===")

    # Calcolare correlazioni
    correlation_matrix = X_train.corr()

    # Identificare features più correlate
    threshold = 0.7
    high_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append(
                    {
                        "feature1": correlation_matrix.columns[i],
                        "feature2": correlation_matrix.columns[j],
                        "correlation": corr_value,
                    }
                )

    logger.info(f"Coppie di features con correlazione > {threshold}:")
    for pair in high_corr_pairs:
        logger.info(
            f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}"
        )

    # Creare heatmap delle correlazioni
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()

    return correlation_matrix, high_corr_pairs


def perform_feature_ablation_study(
    model, X_train, X_test, y_train, y_test, feature_names, top_features=10
):
    """
    Esegue uno studio di ablation per verificare l'impatto delle features più importanti.
    """
    logger.info("=== STUDIO ABLATION FEATURES ===")

    # Selezionare top features
    top_feature_names = feature_names[:top_features]

    ablation_results = []

    # Baseline performance
    baseline_pred = model.predict(X_test)
    baseline_recall = recall_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    logger.info(f"Baseline performance:")
    logger.info(f"  Recall: {baseline_recall:.4f}")
    logger.info(f"  F1-Score: {baseline_f1:.4f}")
    logger.info(f"  Accuracy: {baseline_accuracy:.4f}")

    # Testare rimozione di ogni feature importante
    for feature in top_feature_names:
        # Creare dataset senza la feature
        X_test_ablated = X_test.drop(feature, axis=1)

        # Raddestrare modello senza la feature
        from sklearn.linear_model import LogisticRegression

        ablated_model = LogisticRegression(
            C=0.01,
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        ablated_model.fit(X_train.drop(feature, axis=1), y_train)

        # Valutare performance
        ablated_pred = ablated_model.predict(X_test_ablated)
        ablated_recall = recall_score(y_test, ablated_pred)
        ablated_f1 = f1_score(y_test, ablated_pred)
        ablated_accuracy = accuracy_score(y_test, ablated_pred)

        # Calcolare impatto
        recall_impact = baseline_recall - ablated_recall
        f1_impact = baseline_f1 - ablated_f1
        accuracy_impact = baseline_accuracy - ablated_accuracy

        ablation_results.append(
            {
                "feature": feature,
                "recall_impact": recall_impact,
                "f1_impact": f1_impact,
                "accuracy_impact": accuracy_impact,
                "ablated_recall": ablated_recall,
                "ablated_f1": ablated_f1,
                "ablated_accuracy": ablated_accuracy,
            }
        )

        logger.info(f"Rimozione {feature}:")
        logger.info(f"  Recall impact: {recall_impact:+.4f}")
        logger.info(f"  F1 impact: {f1_impact:+.4f}")
        logger.info(f"  Accuracy impact: {accuracy_impact:+.4f}")

    return pd.DataFrame(ablation_results)


def save_feature_analysis_results(
    coef_df, perm_df, comparison_df, ablation_df, results_dir
):
    """
    Salva tutti i risultati dell'analisi delle features.
    """
    logger.info("=== SALVATAGGIO RISULTATI ===")

    # Salvare DataFrames
    coef_df.to_csv(results_dir / "coefficient_importance.csv", index=False)
    perm_df.to_csv(results_dir / "permutation_importance.csv", index=False)
    comparison_df.to_csv(results_dir / "importance_comparison.csv", index=False)
    ablation_df.to_csv(results_dir / "feature_ablation_study.csv", index=False)

    # Salvare risultati in JSON
    results_summary = {
        "top_5_coefficient_features": coef_df.head(5).to_dict("records"),
        "top_5_permutation_features": perm_df.head(5).to_dict("records"),
        "most_important_feature": coef_df.iloc[0]["feature"],
        "coefficient_importance_score": float(coef_df.iloc[0]["abs_coefficient"]),
        "permutation_importance_score": float(
            perm_df.iloc[0]["permutation_importance"]
        ),
        "total_features_analyzed": len(coef_df),
        "ablation_study_features": len(ablation_df),
    }

    with open(results_dir / "feature_importance_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Risultati salvati in: {results_dir}")


def log_feature_importance_to_mlflow(coef_df, perm_df, ablation_df, experiment_id):
    """
    Logga i risultati dell'analisi delle features in MLflow.
    """
    logger.info("=== LOGGING A MLFLOW ===")

    # Preparare parametri
    params = {
        "analysis_type": "feature_importance",
        "top_features_count": 10,
        "permutation_repeats": 10,
        "ablation_features": len(ablation_df),
    }

    # Preparare metriche
    metrics = {
        "top_coefficient_importance": float(coef_df.iloc[0]["abs_coefficient"]),
        "top_permutation_importance": float(perm_df.iloc[0]["permutation_importance"]),
        "avg_recall_impact": float(ablation_df["recall_impact"].mean()),
        "max_recall_impact": float(ablation_df["recall_impact"].max()),
        "min_recall_impact": float(ablation_df["recall_impact"].min()),
    }

    # Logga in MLflow
    model_name = "feature_importance_analysis"
    artifacts_path = PLOTS_DIR if PLOTS_DIR.exists() else None
    model_details = create_mlflow_run(model_name, params, metrics, artifacts_path, None)

    return model_details


def main():
    """
    Funzione principale per l'analisi dell'importanza delle features.
    """
    logger.info("=== INIZIO FEATURE IMPORTANCE ANALYSIS ===")

    try:
        # 1. Setup MLflow
        experiment_id = setup_mlflow()

        # 2. Caricamento dati e modello
        X_train, X_test, y_train, y_test, model = load_data_and_model()
        feature_names = X_train.columns.tolist()

        # 3. Analisi coefficienti
        coef_df = analyze_coefficient_importance(model, feature_names)

        # 4. Analisi permutation importance
        perm_df = analyze_permutation_importance(model, X_test, y_test, feature_names)

        # 5. Creazione plots
        comparison_df = create_feature_importance_plots(coef_df, perm_df, PLOTS_DIR)

        # 6. Analisi correlazioni
        correlation_matrix, high_corr_pairs = analyze_feature_correlations(
            X_train, feature_names
        )

        # 7. Studio ablation
        ablation_df = perform_feature_ablation_study(
            model, X_train, X_test, y_train, y_test, feature_names
        )

        # 8. Salvare risultati
        save_feature_analysis_results(
            coef_df, perm_df, comparison_df, ablation_df, RESULTS_DIR
        )

        # 9. Logging a MLflow
        mlflow_details = log_feature_importance_to_mlflow(
            coef_df, perm_df, ablation_df, experiment_id
        )

        # 10. Riepilogo finale
        logger.info(f"\n{'='*50}")
        logger.info("🏆 FEATURE IMPORTANCE ANALYSIS COMPLETATA")
        logger.info(f"{'='*50}")
        logger.info(f"📊 Top 5 Features (Coefficient):")
        for i, row in coef_df.head(5).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['coefficient']:.4f}")

        logger.info(f"\n📊 Top 5 Features (Permutation):")
        for i, row in perm_df.head(5).iterrows():
            logger.info(
                f"  {i+1}. {row['feature']}: {row['permutation_importance']:.4f}"
            )

        logger.info(f"\n🎯 Most Important Feature: {coef_df.iloc[0]['feature']}")
        logger.info(f"🎯 Coefficient Value: {coef_df.iloc[0]['coefficient']:.4f}")
        logger.info(
            f"🎯 Permutation Importance: {perm_df.iloc[0]['permutation_importance']:.4f}"
        )
        logger.info(f"{'='*50}")

        return {
            "coefficient_df": coef_df,
            "permutation_df": perm_df,
            "comparison_df": comparison_df,
            "ablation_df": ablation_df,
            "correlation_matrix": correlation_matrix,
            "experiment_id": experiment_id,
        }

    except Exception as e:
        logger.error(f"Errore durante feature importance analysis: {e}")
        raise


if __name__ == "__main__":
    main()
