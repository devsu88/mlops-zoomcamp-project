#!/usr/bin/env python3
"""
Script per scaricare il dataset breast cancer da Kaggle e salvarlo in Google Cloud Storage.
"""

import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from google.cloud import storage

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/wasiqaliyasir/breast-cancer-dataset"
LOCAL_DATA_DIR = Path("data")
BUCKET_NAME = "mlops-breast-cancer-data"
DATASET_NAME = "breast_cancer_dataset.csv"

def download_from_kaggle():
    """
    Download del dataset da Kaggle usando il comando kaggle.
    """
    logger.info("Iniziando download del dataset da Kaggle...")
    
    try:
        # Creare directory locale se non esiste
        LOCAL_DATA_DIR.mkdir(exist_ok=True)
        
        # Cambiare directory per il download
        original_dir = os.getcwd()
        os.chdir(LOCAL_DATA_DIR)
        
        logger.info("Eseguendo comando kaggle...")
        
        # Eseguire il comando kaggle
        cmd = ["kaggle", "datasets", "download", "-d", "wasiqaliyasir/breast-cancer-dataset", "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Tornare alla directory originale
        os.chdir(original_dir)
        
        if result.returncode != 0:
            logger.error(f"Errore nel comando kaggle: {result.stderr}")
            raise Exception(f"Comando kaggle fallito: {result.stderr}")
        
        logger.info("Download completato")
        
        # Trovare il file CSV scaricato
        csv_files = list(LOCAL_DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("Nessun file CSV trovato nel download")
        
        csv_path = csv_files[0]
        logger.info(f"File CSV trovato: {csv_path}")
        
        # Leggere il dataset
        df = pd.read_csv(csv_path)
        
        # Rinominare il file per consistenza
        final_path = LOCAL_DATA_DIR / DATASET_NAME
        df.to_csv(final_path, index=False)
        
        # Pulire file temporanei (mantenere solo il file finale)
        if csv_path != final_path:
            csv_path.unlink()
        
        logger.info(f"Dataset salvato localmente in: {final_path}")
        logger.info(f"Shape del dataset: {df.shape}")
        logger.info(f"Colonne: {list(df.columns)}")
        
        return final_path, df
        
    except Exception as e:
        logger.error(f"Errore durante il download da Kaggle: {e}")
        logger.error("Impossibile scaricare il dataset. Verificare la connessione e l'URL.")
        raise

def upload_to_gcs(local_path: Path, bucket_name: str):
    """
    Upload del dataset a Google Cloud Storage.
    """
    logger.info(f"Uploading dataset a GCS bucket: {bucket_name}")
    
    try:
        # Inizializzare client GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Upload file
        blob_name = f"raw/{DATASET_NAME}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        
        logger.info(f"Dataset caricato in GCS: gs://{bucket_name}/{blob_name}")
        
        return f"gs://{bucket_name}/{blob_name}"
        
    except Exception as e:
        logger.error(f"Errore durante l'upload a GCS: {e}")
        raise

def verify_dataset(df: pd.DataFrame):
    """
    Verifica integrità del dataset.
    """
    logger.info("Verificando integrità del dataset...")
    
    # Pulire il dataset rimuovendo colonne vuote
    df_cleaned = df.copy()
    
    # Rimuovere colonne completamente vuote
    empty_columns = df_cleaned.columns[df_cleaned.isnull().all()].tolist()
    if empty_columns:
        logger.info(f"Rimuovendo colonne vuote: {empty_columns}")
        df_cleaned = df_cleaned.drop(columns=empty_columns)
    
    # Rimuovere colonne con nomi problematici (es. Unnamed)
    unnamed_columns = [col for col in df_cleaned.columns if 'Unnamed' in col]
    if unnamed_columns:
        logger.info(f"Rimuovendo colonne unnamed: {unnamed_columns}")
        df_cleaned = df_cleaned.drop(columns=unnamed_columns)
    
    logger.info(f"Shape dopo pulizia: {df_cleaned.shape}")
    
    # Controlli base
    checks = {
        "Numero di righe": len(df_cleaned) > 0,
        "Numero di colonne": len(df_cleaned.columns) > 0,
        "Missing values": df_cleaned.isnull().sum().sum() == 0,
        "Colonna target presente": 'diagnosis' in df_cleaned.columns,
        "Valori target validi": df_cleaned['diagnosis'].isin(['M', 'B']).all()
    }
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check_name}: {status}")
    
    # Statistiche base
    logger.info(f"Distribuzione target:")
    logger.info(df_cleaned['diagnosis'].value_counts())
    
    # Aggiornare il DataFrame originale
    df.drop(columns=df.columns, inplace=True)
    df[df_cleaned.columns] = df_cleaned
    
    return all(checks.values())

def main():
    """
    Funzione principale per il download e upload del dataset.
    """
    logger.info("=== INIZIO DOWNLOAD DATASET ===")
    
    try:
        # 1. Download dataset
        local_path, df = download_from_kaggle()
        
        # 2. Verifica integrità
        if not verify_dataset(df):
            raise ValueError("Dataset non valido!")
        
        # 3. Upload a GCS
        gcs_path = upload_to_gcs(local_path, BUCKET_NAME)
        
        logger.info("=== DOWNLOAD COMPLETATO CON SUCCESSO ===")
        logger.info(f"Dataset disponibile in: {gcs_path}")
        
        return gcs_path
        
    except Exception as e:
        logger.error(f"Errore durante il processo: {e}")
        raise

if __name__ == "__main__":
    main() 