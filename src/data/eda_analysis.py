#!/usr/bin/env python3
"""
Script per l'Exploratory Data Analysis automatica del dataset breast cancer.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione
warnings.filterwarnings("ignore")
plt.style.use("default")


def load_dataset(data_path: Path):
    """
    Carica il dataset.
    """
    logger.info(f"Caricamento dataset da: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset caricato: {df.shape}")
    return df


def basic_info(df: pd.DataFrame):
    """
    Analisi informazioni base del dataset.
    """
    logger.info("=== ANALISI INFORMAZIONI BASE ===")

    print(f"📊 Shape del dataset: {df.shape}")
    print(f"📋 Colonne: {list(df.columns)}")
    print(f"🎯 Target: {df['diagnosis'].value_counts().to_dict()}")

    # Informazioni base
    print("\n📋 Informazioni del dataset:")
    print(df.info())

    # Statistiche descrittive
    print("\n📊 Statistiche descrittive:")
    print(df.describe())

    return df


def target_analysis(df: pd.DataFrame):
    """
    Analisi del target (diagnosis).
    """
    logger.info("=== ANALISI TARGET ===")

    target_counts = df["diagnosis"].value_counts()
    target_percentages = df["diagnosis"].value_counts(normalize=True) * 100

    print("🎯 Distribuzione del target (diagnosis):")
    print(target_counts)

    print("\n📊 Percentuali:")
    for diagnosis, percentage in target_percentages.items():
        label = "Benigno" if diagnosis == "B" else "Maligno"
        print(f"{label} ({diagnosis}): {percentage:.1f}%")

    # Visualizzazione
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    target_counts.plot(kind="bar", color=["lightblue", "lightcoral"])
    plt.title("Distribuzione Target")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Benigno (B)", "Maligno (M)"])

    plt.subplot(1, 2, 2)
    plt.pie(
        target_counts.values,
        labels=["Benigno (B)", "Maligno (M)"],
        autopct="%1.1f%%",
        colors=["lightblue", "lightcoral"],
    )
    plt.title("Distribuzione Target (%)")

    plt.tight_layout()
    plt.savefig("notebooks/target_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    return target_counts


def missing_values_analysis(df: pd.DataFrame):
    """
    Analisi missing values.
    """
    logger.info("=== ANALISI MISSING VALUES ===")

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_df = pd.DataFrame(
        {"Missing_Count": missing_values, "Missing_Percentage": missing_percentage}
    )

    print("🔍 Analisi Missing Values:")
    missing_columns = missing_df[missing_df["Missing_Count"] > 0]

    if len(missing_columns) == 0:
        print("✅ Nessun missing value trovato!")
    else:
        print(missing_columns)
        print(f"⚠️ Trovati {missing_df['Missing_Count'].sum()} missing values totali")

    return missing_df


def correlation_analysis(df: pd.DataFrame):
    """
    Analisi correlazioni.
    """
    logger.info("=== ANALISI CORRELAZIONI ===")

    # Features numeriche
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 Features numeriche ({len(numerical_features)}): {numerical_features}")

    # Matrice di correlazione
    correlation_matrix = df[numerical_features].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matrice di Correlazione - Features Numeriche", fontsize=16)
    plt.tight_layout()
    plt.savefig("notebooks/correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Correlazioni con il target
    df_with_target = df.copy()
    df_with_target["diagnosis_numeric"] = (df_with_target["diagnosis"] == "M").astype(
        int
    )

    target_correlations = (
        df_with_target[numerical_features + ["diagnosis_numeric"]]
        .corr()["diagnosis_numeric"]
        .sort_values(ascending=False)
    )

    print("\n🎯 Top 10 Correlazioni con il target (diagnosis):")
    top_correlations = target_correlations.drop("diagnosis_numeric").head(10)
    for feature, corr in top_correlations.items():
        print(f"  {feature}: {corr:.3f}")

    return target_correlations


def outliers_analysis(df: pd.DataFrame):
    """
    Analisi outliers.
    """
    logger.info("=== ANALISI OUTLIERS ===")

    features_to_analyze = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
    ]

    print("🔍 Analisi Outliers (IQR method):")
    outliers_summary = {}

    for feature in features_to_analyze:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outliers_count = len(outliers)
        outliers_percentage = outliers_count / len(df) * 100

        outliers_summary[feature] = {
            "count": outliers_count,
            "percentage": outliers_percentage,
        }

        print(f"{feature}: {outliers_count} outliers ({outliers_percentage:.1f}%)")

    # Box plots
    plt.figure(figsize=(20, 12))
    for i, feature in enumerate(features_to_analyze, 1):
        plt.subplot(2, 4, i)
        sns.boxplot(data=df, x="diagnosis", y=feature)
        plt.title(f"{feature} vs Diagnosis")
        plt.xlabel("Diagnosis")
        plt.ylabel(feature)

    plt.tight_layout()
    plt.savefig("notebooks/outliers_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    return outliers_summary


def feature_distributions(df: pd.DataFrame):
    """
    Analisi distribuzioni features.
    """
    logger.info("=== ANALISI DISTRIBUZIONI FEATURES ===")

    features_to_plot = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
    ]

    plt.figure(figsize=(20, 12))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 4, i)

        # Plot per ogni classe
        for diagnosis in ["B", "M"]:
            subset = df[df["diagnosis"] == diagnosis][feature]
            plt.hist(subset, alpha=0.7, label=f"Diagnosis {diagnosis}", bins=20)

        plt.title(f"Distribuzione {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.savefig("notebooks/feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_eda_report(df: pd.DataFrame, target_correlations, outliers_summary):
    """
    Genera report finale EDA.
    """
    logger.info("=== GENERAZIONE REPORT EDA ===")

    report = f"""
# EDA Report - Breast Cancer Dataset

## 📊 Dataset Overview
- **Shape**: {df.shape}
- **Target Distribution**: {dict(df['diagnosis'].value_counts())}
- **Numerical Features**: {len(df.select_dtypes(include=[np.number]).columns)}
- **Missing Values**: {df.isnull().sum().sum()}

## 🎯 Target Analysis
- **Benigno (B)**: {df['diagnosis'].value_counts()['B']} ({df['diagnosis'].value_counts(normalize=True)['B']*100:.1f}%)
- **Maligno (M)**: {df['diagnosis'].value_counts()['M']} ({df['diagnosis'].value_counts(normalize=True)['M']*100:.1f}%)

## 🏆 Top 5 Features per Correlazione con Target
"""

    top_features = target_correlations.drop("diagnosis_numeric").head(5)
    for feature, corr in top_features.items():
        report += f"- **{feature}**: {corr:.3f}\n"

    report += f"""
## 🔍 Outliers Summary
"""

    for feature, info in outliers_summary.items():
        report += (
            f"- **{feature}**: {info['count']} outliers ({info['percentage']:.1f}%)\n"
        )

    report += f"""
## ✅ Conclusioni
1. Dataset bilanciato con leggera prevalenza di casi benigni
2. Nessun missing value presente
3. Features fortemente correlate con il target identificate
4. Presenza di outliers in alcune features (normale per dati medici)
5. Dataset pronto per il preprocessing e modeling

## 📈 Prossimi Passi
1. Preprocessing delle features
2. Feature selection basata su correlazioni
3. Train/test split
4. Model development
"""

    # Salvare report
    with open("notebooks/eda_report.md", "w") as f:
        f.write(report)

    print("📋 Report EDA generato: notebooks/eda_report.md")
    print(report)

    return report


def main():
    """
    Funzione principale per l'EDA.
    """
    logger.info("=== INIZIO EDA BREAST CANCER DATASET ===")

    try:
        # 1. Caricamento dataset
        data_path = Path("data/breast_cancer_dataset.csv")
        df = load_dataset(data_path)

        # 2. Analisi base
        df = basic_info(df)

        # 3. Analisi target
        target_counts = target_analysis(df)

        # 4. Analisi missing values
        missing_df = missing_values_analysis(df)

        # 5. Analisi correlazioni
        target_correlations = correlation_analysis(df)

        # 6. Analisi outliers
        outliers_summary = outliers_analysis(df)

        # 7. Distribuzioni features
        feature_distributions(df)

        # 8. Generazione report
        report = generate_eda_report(df, target_correlations, outliers_summary)

        logger.info("=== EDA COMPLETATA CON SUCCESSO ===")

        return df, target_correlations, outliers_summary

    except Exception as e:
        logger.error(f"Errore durante l'EDA: {e}")
        raise


if __name__ == "__main__":
    main()
