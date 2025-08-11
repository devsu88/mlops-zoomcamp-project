#!/usr/bin/env python3
"""
Script per il preprocessing del dataset breast cancer.
Include pulizia dati, feature engineering, scaling e split train/test.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurazione
LOCAL_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = Path("data/processed")
BUCKET_NAME = "mlops-breast-cancer-data"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw_dataset(data_path: Path):
    """
    Carica il dataset grezzo.
    """
    logger.info(f"Caricamento dataset da: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset caricato: {df.shape}")
    return df


def clean_dataset(df: pd.DataFrame):
    """
    Pulizia del dataset: rimozione colonne vuote e duplicati.
    """
    logger.info("=== PULIZIA DATASET ===")

    # Shape iniziale
    initial_shape = df.shape
    logger.info(f"Shape iniziale: {initial_shape}")

    # Rimuovere colonne completamente vuote
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        logger.info(f"Rimuovendo colonne vuote: {empty_columns}")
        df = df.drop(columns=empty_columns)

    # Rimuovere colonne con nomi problematici (es. Unnamed)
    unnamed_columns = [col for col in df.columns if "Unnamed" in col]
    if unnamed_columns:
        logger.info(f"Rimuovendo colonne unnamed: {unnamed_columns}")
        df = df.drop(columns=unnamed_columns)

    # Rimuovere duplicati
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Rimuovendo {duplicates} righe duplicate")
        df = df.drop_duplicates()

    # Shape finale
    final_shape = df.shape
    logger.info(f"Shape dopo pulizia: {final_shape}")
    logger.info(f"Righe rimosse: {initial_shape[0] - final_shape[0]}")
    logger.info(f"Colonne rimosse: {initial_shape[1] - final_shape[1]}")

    return df


def feature_engineering(df: pd.DataFrame):
    """
    Feature engineering: creazione nuove features e trasformazioni.
    """
    logger.info("=== FEATURE ENGINEERING ===")

    df_processed = df.copy()

    # Separare features per categoria
    mean_features = [
        col for col in df_processed.columns if "mean" in col and col != "diagnosis"
    ]
    se_features = [col for col in df_processed.columns if "_se" in col]
    worst_features = [col for col in df_processed.columns if "worst" in col]

    logger.info(f"Features MEAN: {len(mean_features)}")
    logger.info(f"Features SE: {len(se_features)}")
    logger.info(f"Features WORST: {len(worst_features)}")

    # Creare features aggregate
    if mean_features:
        df_processed["mean_features_avg"] = df_processed[mean_features].mean(axis=1)
        df_processed["mean_features_std"] = df_processed[mean_features].std(axis=1)

    if worst_features:
        df_processed["worst_features_avg"] = df_processed[worst_features].mean(axis=1)
        df_processed["worst_features_max"] = df_processed[worst_features].max(axis=1)

    # Creare ratios delle features più importanti (da EDA)
    if "radius_mean" in df_processed.columns and "radius_worst" in df_processed.columns:
        df_processed["radius_ratio"] = (
            df_processed["radius_worst"] / df_processed["radius_mean"]
        )

    if (
        "perimeter_mean" in df_processed.columns
        and "perimeter_worst" in df_processed.columns
    ):
        df_processed["perimeter_ratio"] = (
            df_processed["perimeter_worst"] / df_processed["perimeter_mean"]
        )

    if "area_mean" in df_processed.columns and "area_worst" in df_processed.columns:
        df_processed["area_ratio"] = (
            df_processed["area_worst"] / df_processed["area_mean"]
        )

    logger.info(
        f"Features aggiunte durante feature engineering: {df_processed.shape[1] - df.shape[1]}"
    )
    logger.info(f"Shape dopo feature engineering: {df_processed.shape}")

    return df_processed


def encode_target(df: pd.DataFrame, target_column: str = "diagnosis"):
    """
    Encoding del target da categorico a numerico.
    """
    logger.info("=== ENCODING TARGET ===")

    df_encoded = df.copy()

    # Encoding: M=1 (Maligno), B=0 (Benigno)
    label_encoder = LabelEncoder()
    df_encoded[target_column + "_encoded"] = label_encoder.fit_transform(
        df_encoded[target_column]
    )

    # Mapping per referenza
    mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    logger.info(f"Target encoding mapping: {mapping}")

    # Verifica distribuzione
    target_distribution = (
        df_encoded[target_column + "_encoded"].value_counts().sort_index()
    )
    logger.info(f"Target distribution: {target_distribution.to_dict()}")

    return df_encoded, label_encoder


def select_features(X: pd.DataFrame, y: pd.Series, k_best: int = 20):
    """
    Feature selection usando SelectKBest.
    """
    logger.info("=== FEATURE SELECTION ===")

    # Rimuovere features non numeriche (eccetto id se presente)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if "id" in numeric_features:
        numeric_features.remove("id")

    X_numeric = X[numeric_features]
    logger.info(f"Features numeriche per selection: {len(numeric_features)}")

    # SelectKBest con f_classif
    selector = SelectKBest(score_func=f_classif, k=min(k_best, len(numeric_features)))
    X_selected = selector.fit_transform(X_numeric, y)

    # Ottenere nomi features selezionate
    selected_features = X_numeric.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]

    logger.info(
        f"Features selezionate: {len(selected_features)} su {len(numeric_features)}"
    )

    # Log delle top features
    feature_ranking = list(zip(selected_features, feature_scores))
    feature_ranking.sort(key=lambda x: x[1], reverse=True)

    logger.info("Top 10 features selezionate:")
    for feature, score in feature_ranking[:10]:
        logger.info(f"  {feature}: {score:.2f}")

    return X_selected, selected_features, selector


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Scaling delle features usando StandardScaler.
    """
    logger.info("=== SCALING FEATURES ===")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        f"Features scalate - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}"
    )

    return X_train_scaled, X_test_scaled, scaler


def train_test_split_data(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split del dataset in train e test.
    """
    logger.info("=== TRAIN/TEST SPLIT ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(
        f"Train target distribution: {pd.Series(y_train).value_counts().to_dict()}"
    )
    logger.info(
        f"Test target distribution: {pd.Series(y_test).value_counts().to_dict()}"
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    selected_features,
    scaler,
    label_encoder,
    selector,
):
    """
    Salvare i dati processati e i transformers.
    """
    logger.info("=== SALVATAGGIO DATI PROCESSATI ===")

    # Creare directory se non esistente
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    # Salvare arrays come DataFrames per preservare struttura
    train_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    train_df["target"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
    test_df["target"] = y_test.reset_index(drop=True)

    # Salvare datasets
    train_path = PROCESSED_DATA_DIR / "train_set.csv"
    test_path = PROCESSED_DATA_DIR / "test_set.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set salvato: {train_path}")
    logger.info(f"Test set salvato: {test_path}")

    # Salvare transformers
    scaler_path = PROCESSED_DATA_DIR / "scaler.joblib"
    encoder_path = PROCESSED_DATA_DIR / "label_encoder.joblib"
    selector_path = PROCESSED_DATA_DIR / "feature_selector.joblib"

    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(selector, selector_path)

    logger.info(f"Scaler salvato: {scaler_path}")
    logger.info(f"Label encoder salvato: {encoder_path}")
    logger.info(f"Feature selector salvato: {selector_path}")

    # Salvare metadati
    metadata = {
        "dataset_shape": int(train_df.shape[0] + test_df.shape[0]),
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
        "selected_features": selected_features,
        "test_size": float(TEST_SIZE),
        "random_state": int(RANDOM_STATE),
        "target_mapping": {
            str(k): int(v)
            for k, v in zip(
                label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
            )
        },
    }

    metadata_path = PROCESSED_DATA_DIR / "preprocessing_metadata.json"
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata salvati: {metadata_path}")

    return train_path, test_path


def upload_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str = "processed"):
    """
    Upload dei dati processati a Google Cloud Storage.
    """
    logger.info("=== UPLOAD A GCS ===")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        for file_path in local_dir.glob("*"):
            if file_path.is_file():
                blob_name = f"{gcs_prefix}/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded: gs://{bucket_name}/{blob_name}")

        logger.info("Upload completato con successo!")

    except Exception as e:
        logger.error(f"Errore durante upload a GCS: {e}")
        logger.info("Continuando con files locali...")


def main():
    """
    Funzione principale per il preprocessing.
    """
    logger.info("=== INIZIO PREPROCESSING BREAST CANCER DATASET ===")

    try:
        # 1. Caricamento dataset
        data_path = LOCAL_DATA_DIR / "breast_cancer_dataset.csv"
        df = load_raw_dataset(data_path)

        # 2. Pulizia dataset
        df_clean = clean_dataset(df)

        # 3. Feature engineering
        df_engineered = feature_engineering(df_clean)

        # 4. Encoding target
        df_encoded, label_encoder = encode_target(df_engineered)

        # 5. Preparazione X e y
        target_column = "diagnosis_encoded"
        feature_columns = [
            col
            for col in df_encoded.columns
            if col not in ["diagnosis", "diagnosis_encoded"]
        ]

        X = df_encoded[feature_columns]
        y = df_encoded[target_column]

        # 6. Feature selection
        X_selected, selected_features, selector = select_features(X, y, k_best=20)

        # 7. Train/test split
        X_train, X_test, y_train, y_test = train_test_split_data(X_selected, y)

        # 8. Scaling
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # 9. Salvataggio
        train_path, test_path = save_processed_data(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            selected_features,
            scaler,
            label_encoder,
            selector,
        )

        # 10. Upload a GCS
        upload_to_gcs(PROCESSED_DATA_DIR, BUCKET_NAME)

        logger.info("=== PREPROCESSING COMPLETATO CON SUCCESSO ===")
        logger.info(f"Train set: {train_path}")
        logger.info(f"Test set: {test_path}")
        logger.info(f"Transformers salvati in: {PROCESSED_DATA_DIR}")

        return {
            "train_path": train_path,
            "test_path": test_path,
            "selected_features": selected_features,
            "train_shape": X_train_scaled.shape,
            "test_shape": X_test_scaled.shape,
        }

    except Exception as e:
        logger.error(f"Errore durante il preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
