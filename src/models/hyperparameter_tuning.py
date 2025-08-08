#!/usr/bin/env python3
"""
Script per hyperparameter tuning del Logistic Regression.
Include GridSearch, RandomSearch e MLflow tracking.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold)

from .mlflow_config import create_mlflow_run, setup_mlflow

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione
PROCESSED_DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
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
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def define_hyperparameter_grids():
    """
    Definisce i grid di hyperparameter per Logistic Regression.
    """
    logger.info("=== DEFINIZIONE HYPERPARAMETER GRIDS ===")
    
    # Grid per GridSearch
    grid_params = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000],
        'class_weight': ['balanced', None]
    }
    
    # Grid per RandomSearch (più ampio)
    random_params = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Per elasticnet
    }
    
    logger.info(f"Grid parameters: {len(grid_params['C']) * len(grid_params['penalty']) * len(grid_params['solver']) * len(grid_params['max_iter']) * len(grid_params['class_weight'])} combinations")
    logger.info(f"Random parameters: {len(random_params['C']) * len(random_params['penalty']) * len(random_params['solver']) * len(random_params['max_iter']) * len(random_params['class_weight']) * len(random_params['l1_ratio'])} combinations")
    
    return grid_params, random_params

def perform_grid_search(X_train, y_train, grid_params):
    """
    Esegue GridSearch per Logistic Regression.
    """
    logger.info("=== GRID SEARCH LOGISTIC REGRESSION ===")
    
    # Base model
    base_model = LogisticRegression(random_state=RANDOM_STATE)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # GridSearch con Recall come scoring (priorità massima)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=grid_params,
        cv=cv,
        scoring='recall',  # Priorità massima per il nostro caso
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Iniziando GridSearch...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV Recall: {grid_search.best_score_:.4f}")
    
    return grid_search

def perform_random_search(X_train, y_train, random_params, n_iter=50):
    """
    Esegue RandomSearch per Logistic Regression.
    """
    logger.info("=== RANDOM SEARCH LOGISTIC REGRESSION ===")
    
    # Base model
    base_model = LogisticRegression(random_state=RANDOM_STATE)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # RandomSearch con Recall come scoring
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=random_params,
        n_iter=n_iter,
        cv=cv,
        scoring='recall',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Iniziando RandomSearch con {n_iter} iterations...")
    random_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV Recall: {random_search.best_score_:.4f}")
    
    return random_search

def evaluate_tuned_model(model, X_train, X_test, y_train, y_test, search_name):
    """
    Valuta il modello ottimizzato.
    """
    logger.info(f"=== VALUTAZIONE {search_name.upper()} ===")
    
    # Predizioni
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Cross-validation scores
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recall = model.best_score_
    
    # Calcolare F1 e accuracy manualmente se non disponibili
    try:
        cv_f1 = model.cv_results_['mean_test_f1'][model.best_index_]
    except KeyError:
        # Calcolare F1 manualmente
        cv_f1_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.best_estimator_.fit(X_cv_train, y_cv_train)
            y_cv_pred = model.best_estimator_.predict(X_cv_val)
            cv_f1_scores.append(f1_score(y_cv_val, y_cv_pred))
        cv_f1 = np.mean(cv_f1_scores)
    
    try:
        cv_accuracy = model.cv_results_['mean_test_accuracy'][model.best_index_]
    except KeyError:
        # Calcolare accuracy manualmente
        cv_accuracy_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.best_estimator_.fit(X_cv_train, y_cv_train)
            y_cv_pred = model.best_estimator_.predict(X_cv_val)
            cv_accuracy_scores.append(accuracy_score(y_cv_val, y_cv_pred))
        cv_accuracy = np.mean(cv_accuracy_scores)
    
    results = {
        'search_name': search_name,
        'best_params': model.best_params_,
        'cv_recall': cv_recall,
        'cv_f1': cv_f1,
        'cv_accuracy': cv_accuracy,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_roc_auc': roc_auc,
        'test_pr_auc': pr_auc,
        'n_cv_splits': CV_FOLDS,
        'n_iterations': getattr(model, 'n_iter', len(model.cv_results_['params']))
    }
    
    logger.info(f"Test Results - {search_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")
    
    return results, model.best_estimator_

def log_tuning_results_to_mlflow(results, model, search_name):
    """
    Logga i risultati del tuning in MLflow.
    """
    logger.info(f"=== LOGGING {search_name.upper()} A MLFLOW ===")
    
    # Preparare parametri per MLflow
    params = results['best_params'].copy()
    params['search_method'] = search_name
    params['cv_folds'] = CV_FOLDS
    params['random_state'] = RANDOM_STATE
    
    # Preparare metriche per MLflow
    metrics = {
        'cv_recall': results['cv_recall'],
        'cv_f1': results['cv_f1'],
        'cv_accuracy': results['cv_accuracy'],
        'test_accuracy': results['test_accuracy'],
        'test_precision': results['test_precision'],
        'test_recall': results['test_recall'],
        'test_f1': results['test_f1'],
        'test_roc_auc': results['test_roc_auc'],
        'test_pr_auc': results['test_pr_auc'],
        'n_iterations': results['n_iterations']
    }
    
    # Logga in MLflow
    model_name = f"logistic_regression_{search_name.lower()}"
    artifacts_path = RESULTS_DIR / "plots" if (RESULTS_DIR / "plots").exists() else None
    model_details = create_mlflow_run(model_name, params, metrics, artifacts_path, model)
    
    return model_details

def save_tuned_model(model, search_name, results_dir, results):
    """
    Salva il modello ottimizzato.
    """
    results_dir.mkdir(exist_ok=True)
    model_path = results_dir / f"logistic_regression_{search_name.lower()}_tuned.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Modello {search_name} salvato: {model_path}")
    
    # Salvare anche i risultati
    results_path = results_dir / f"logistic_regression_{search_name.lower()}_tuning_results.json"
    
    import json
    results_json = {
        'search_name': results['search_name'],
        'best_params': results['best_params'],
        'cv_recall': float(results['cv_recall']),
        'cv_f1': float(results['cv_f1']),
        'cv_accuracy': float(results['cv_accuracy']),
        'test_accuracy': float(results['test_accuracy']),
        'test_precision': float(results['test_precision']),
        'test_recall': float(results['test_recall']),
        'test_f1': float(results['test_f1']),
        'test_roc_auc': float(results['test_roc_auc']),
        'test_pr_auc': float(results['test_pr_auc']),
        'n_cv_splits': results['n_cv_splits'],
        'n_iterations': results['n_iterations']
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Risultati {search_name} salvati: {results_path}")

def create_tuning_comparison_report(all_results, results_dir):
    """
    Crea un report di confronto tra i metodi di tuning.
    """
    logger.info("=== CREAZIONE REPORT CONFRONTO TUNING ===")
    
    comparison_data = []
    for search_name, results in all_results.items():
        comparison_data.append({
            'Search_Method': search_name,
            'Best_CV_Recall': results['cv_recall'],
            'Best_CV_F1': results['cv_f1'],
            'Best_CV_Accuracy': results['cv_accuracy'],
            'Test_Recall': results['test_recall'],
            'Test_F1': results['test_f1'],
            'Test_Precision': results['test_precision'],
            'Test_Accuracy': results['test_accuracy'],
            'Test_ROC_AUC': results['test_roc_auc'],
            'Test_PR_AUC': results['test_pr_auc'],
            'N_Iterations': results['n_iterations'],
            'Best_Params': str(results['best_params'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Salvare report
    report_path = results_dir / "hyperparameter_tuning_comparison.csv"
    comparison_df.to_csv(report_path, index=False)
    logger.info(f"Report tuning salvato: {report_path}")
    
    # Log del confronto
    logger.info("\n=== CONFRONTO HYPERPARAMETER TUNING ===")
    logger.info(comparison_df.to_string(index=False))
    
    return comparison_df

def main():
    """
    Funzione principale per hyperparameter tuning.
    """
    logger.info("=== INIZIO HYPERPARAMETER TUNING ===")
    
    try:
        # 1. Setup MLflow
        experiment_id = setup_mlflow()
        
        # 2. Caricamento dati
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # 3. Definizione grid di parametri
        grid_params, random_params = define_hyperparameter_grids()
        
        # 4. GridSearch
        logger.info("\n" + "="*50)
        grid_search = perform_grid_search(X_train, y_train, grid_params)
        grid_results, grid_model = evaluate_tuned_model(
            grid_search, X_train, X_test, y_train, y_test, "GridSearch"
        )
        
        # 5. RandomSearch
        logger.info("\n" + "="*50)
        random_search = perform_random_search(X_train, y_train, random_params, n_iter=50)
        random_results, random_model = evaluate_tuned_model(
            random_search, X_train, X_test, y_train, y_test, "RandomSearch"
        )
        
        # 6. Logging a MLflow
        logger.info("\n" + "="*50)
        grid_mlflow = log_tuning_results_to_mlflow(grid_results, grid_model, "GridSearch")
        random_mlflow = log_tuning_results_to_mlflow(random_results, random_model, "RandomSearch")
        
        # 7. Salvare modelli
        save_tuned_model(grid_model, "GridSearch", RESULTS_DIR, grid_results)
        save_tuned_model(random_model, "RandomSearch", RESULTS_DIR, random_results)
        
        # 8. Report di confronto
        all_results = {
            'GridSearch': grid_results,
            'RandomSearch': random_results
        }
        comparison_df = create_tuning_comparison_report(all_results, RESULTS_DIR)
        
        # 9. Selezione best model
        best_method = comparison_df.loc[comparison_df['Test_Recall'].idxmax(), 'Search_Method']
        best_recall = comparison_df.loc[comparison_df['Test_Recall'].idxmax(), 'Test_Recall']
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🏆 BEST TUNED MODEL: {best_method}")
        logger.info(f"🏆 BEST RECALL: {best_recall:.4f}")
        logger.info(f"{'='*50}")
        
        logger.info("=== HYPERPARAMETER TUNING COMPLETATO ===")
        
        return {
            'best_method': best_method,
            'best_recall': best_recall,
            'comparison_df': comparison_df,
            'grid_results': grid_results,
            'random_results': random_results,
            'experiment_id': experiment_id
        }
        
    except Exception as e:
        logger.error(f"Errore durante hyperparameter tuning: {e}")
        raise

if __name__ == "__main__":
    main() 