#!/usr/bin/env python3
"""
Script per la validazione approfondita del modello ottimizzato.
Include cross-validation, bootstrap validation, learning curves e analisi degli errori.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     learning_curve, validation_curve)
from sklearn.utils import resample

from .mlflow_config import create_mlflow_run, setup_mlflow

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione
PROCESSED_DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RANDOM_STATE = 42
CV_FOLDS = 5

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
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Caricare modello ottimizzato
    model_path = RESULTS_DIR / "logistic_regression_gridsearch_tuned.joblib"
    model = joblib.load(model_path)
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Modello caricato: {model_path}")
    
    return X_train, X_test, y_train, y_test, model

def perform_cross_validation(model, X_train, y_train):
    """
    Esegue cross-validation approfondita.
    """
    logger.info("=== CROSS-VALIDATION APPROFONDITA ===")
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Metriche da valutare
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = {}
    
    for metric_name, metric_scorer in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric_scorer)
        cv_results[metric_name] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        logger.info(f"{metric_name.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
        logger.info(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return cv_results

def perform_bootstrap_validation(model, X_test, y_test, n_bootstrap=1000):
    """
    Esegue bootstrap validation per stimare intervalli di confidenza.
    """
    logger.info("=== BOOTSTRAP VALIDATION ===")
    
    bootstrap_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        X_boot, y_boot = resample(X_test, y_test, random_state=RANDOM_STATE + i)
        
        # Predizioni
        y_pred = model.predict(X_boot)
        y_pred_proba = model.predict_proba(X_boot)[:, 1]
        
        # Calcolare metriche
        bootstrap_results['accuracy'].append(accuracy_score(y_boot, y_pred))
        bootstrap_results['precision'].append(precision_score(y_boot, y_pred))
        bootstrap_results['recall'].append(recall_score(y_boot, y_pred))
        bootstrap_results['f1'].append(f1_score(y_boot, y_pred))
        bootstrap_results['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba))
        bootstrap_results['pr_auc'].append(average_precision_score(y_boot, y_pred_proba))
    
    # Calcolare statistiche
    bootstrap_stats = {}
    for metric, values in bootstrap_results.items():
        values = np.array(values)
        bootstrap_stats[metric] = {
            'mean': values.mean(),
            'std': values.std(),
            'ci_95_lower': np.percentile(values, 2.5),
            'ci_95_upper': np.percentile(values, 97.5),
            'ci_99_lower': np.percentile(values, 0.5),
            'ci_99_upper': np.percentile(values, 99.5)
        }
        
        logger.info(f"{metric.upper()}: {values.mean():.4f} ± {values.std():.4f}")
        logger.info(f"  95% CI: [{bootstrap_stats[metric]['ci_95_lower']:.4f}, {bootstrap_stats[metric]['ci_95_upper']:.4f}]")
        logger.info(f"  99% CI: [{bootstrap_stats[metric]['ci_99_lower']:.4f}, {bootstrap_stats[metric]['ci_99_upper']:.4f}]")
    
    return bootstrap_results, bootstrap_stats

def analyze_learning_curves(model, X_train, y_train):
    """
    Analizza le learning curves per verificare overfitting/underfitting.
    """
    logger.info("=== ANALISI LEARNING CURVES ===")
    
    # Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='recall',
        n_jobs=-1
    )
    
    # Calcolare medie e deviazioni standard
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    logger.info("Learning curves analysis:")
    logger.info(f"  Final train score: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    logger.info(f"  Final validation score: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    logger.info(f"  Gap (overfitting indicator): {train_mean[-1] - val_mean[-1]:.4f}")
    
    return {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std
    }

def analyze_validation_curves(model, X_train, y_train):
    """
    Analizza le validation curves per i parametri principali.
    """
    logger.info("=== ANALISI VALIDATION CURVES ===")
    
    # Validation curve per C (regularizzazione)
    param_range = np.logspace(-4, 2, 20)
    train_scores, val_scores = validation_curve(
        model, X_train, y_train,
        param_name='C',
        param_range=param_range,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='recall',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    logger.info(f"Best C value: {param_range[np.argmax(val_mean)]:.4f}")
    logger.info(f"Best validation score: {val_mean.max():.4f}")
    
    return {
        'param_range': param_range,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std
    }

def analyze_error_analysis(model, X_test, y_test):
    """
    Analizza gli errori del modello per identificare pattern.
    """
    logger.info("=== ANALISI ERRORI ===")
    
    # Predizioni
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Identificare errori
    errors = X_test[y_test != y_pred].copy()
    error_labels = y_test[y_test != y_pred]
    error_predictions = y_pred[y_test != y_pred]
    error_probabilities = y_pred_proba[y_test != y_pred]
    
    logger.info(f"Errori totali: {len(errors)}")
    logger.info(f"False Positives: {np.sum((y_test == 0) & (y_pred == 1))}")
    logger.info(f"False Negatives: {np.sum((y_test == 1) & (y_pred == 0))}")
    
    # Analizzare caratteristiche degli errori
    if len(errors) > 0:
        error_analysis = {
            'error_count': len(errors),
            'false_positives': int(np.sum((y_test == 0) & (y_pred == 1))),
            'false_negatives': int(np.sum((y_test == 1) & (y_pred == 0))),
            'avg_error_confidence': float(np.mean(error_probabilities)),
            'error_features_mean': errors.mean().to_dict(),
            'error_features_std': errors.std().to_dict()
        }
        
        logger.info("Caratteristiche degli errori:")
        logger.info(f"  Confidenza media: {error_analysis['avg_error_confidence']:.4f}")
        logger.info(f"  False Positives: {error_analysis['false_positives']}")
        logger.info(f"  False Negatives: {error_analysis['false_negatives']}")
        
        # Top features negli errori
        error_feature_importance = errors.std().sort_values(ascending=False)
        logger.info("Top 5 features con più varianza negli errori:")
        for feature, std_val in error_feature_importance.head(5).items():
            logger.info(f"  {feature}: {std_val:.4f}")
    
    return {
        'confusion_matrix': cm,
        'errors': errors,
        'error_labels': error_labels,
        'error_predictions': error_predictions,
        'error_probabilities': error_probabilities,
        'error_analysis': error_analysis if len(errors) > 0 else None
    }

def create_validation_plots(cv_results, bootstrap_results, learning_curves, validation_curves, error_analysis, plots_dir):
    """
    Crea visualizzazioni per la validazione del modello.
    """
    logger.info("=== CREAZIONE PLOTS VALIDAZIONE ===")
    
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Cross-validation results
    plt.figure(figsize=(12, 8))
    metrics = list(cv_results.keys())
    means = [cv_results[m]['mean'] for m in metrics]
    stds = [cv_results[m]['std'] for m in metrics]
    
    x = np.arange(len(metrics))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Cross-Validation Results')
    plt.xticks(x, [m.upper() for m in metrics])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bootstrap confidence intervals
    plt.figure(figsize=(14, 10))
    
    bootstrap_stats = {}
    for metric, values in bootstrap_results.items():
        values = np.array(values)
        bootstrap_stats[metric] = {
            'mean': values.mean(),
            'std': values.std(),
            'ci_95_lower': np.percentile(values, 2.5),
            'ci_95_upper': np.percentile(values, 97.5)
        }
    
    metrics = list(bootstrap_stats.keys())
    means = [bootstrap_stats[m]['mean'] for m in metrics]
    ci_lower = [bootstrap_stats[m]['ci_95_lower'] for m in metrics]
    ci_upper = [bootstrap_stats[m]['ci_95_upper'] for m in metrics]
    
    x = np.arange(len(metrics))
    plt.bar(x, means, yerr=[np.array(means) - np.array(ci_lower), np.array(ci_upper) - np.array(means)], 
            capsize=5, alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Bootstrap Validation Results (95% CI)')
    plt.xticks(x, [m.upper() for m in metrics])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'bootstrap_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Learning curves
    plt.figure(figsize=(12, 8))
    
    train_sizes = learning_curves['train_sizes']
    train_mean = learning_curves['train_mean']
    train_std = learning_curves['train_std']
    val_mean = learning_curves['val_mean']
    val_std = learning_curves['val_std']
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Recall Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Validation curves
    plt.figure(figsize=(12, 8))
    
    param_range = validation_curves['param_range']
    train_mean = validation_curves['train_mean']
    train_std = validation_curves['train_std']
    val_mean = validation_curves['val_mean']
    val_std = validation_curves['val_std']
    
    plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.semilogx(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Recall Score')
    plt.title('Validation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'validation_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Confusion matrix
    if error_analysis['confusion_matrix'] is not None:
        plt.figure(figsize=(8, 6))
        cm = error_analysis['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots salvati in: {plots_dir}")

def save_validation_results(cv_results, bootstrap_stats, learning_curves, validation_curves, error_analysis, results_dir):
    """
    Salva tutti i risultati della validazione.
    """
    logger.info("=== SALVATAGGIO RISULTATI VALIDAZIONE ===")
    
    # Salvare risultati cross-validation
    cv_df = pd.DataFrame({
        'metric': list(cv_results.keys()),
        'mean': [cv_results[m]['mean'] for m in cv_results.keys()],
        'std': [cv_results[m]['std'] for m in cv_results.keys()],
        'min': [cv_results[m]['min'] for m in cv_results.keys()],
        'max': [cv_results[m]['max'] for m in cv_results.keys()]
    })
    cv_df.to_csv(results_dir / 'cross_validation_results.csv', index=False)
    
    # Salvare risultati bootstrap
    bootstrap_df = pd.DataFrame({
        'metric': list(bootstrap_stats.keys()),
        'mean': [bootstrap_stats[m]['mean'] for m in bootstrap_stats.keys()],
        'std': [bootstrap_stats[m]['std'] for m in bootstrap_stats.keys()],
        'ci_95_lower': [bootstrap_stats[m]['ci_95_lower'] for m in bootstrap_stats.keys()],
        'ci_95_upper': [bootstrap_stats[m]['ci_95_upper'] for m in bootstrap_stats.keys()],
        'ci_99_lower': [bootstrap_stats[m]['ci_99_lower'] for m in bootstrap_stats.keys()],
        'ci_99_upper': [bootstrap_stats[m]['ci_99_upper'] for m in bootstrap_stats.keys()]
    })
    bootstrap_df.to_csv(results_dir / 'bootstrap_validation_results.csv', index=False)
    
    # Salvare riepilogo in JSON
    validation_summary = {
        'cross_validation': {
            'folds': CV_FOLDS,
            'results': {
                metric: {
                    'mean': float(cv_results[metric]['mean']),
                    'std': float(cv_results[metric]['std']),
                    'min': float(cv_results[metric]['min']),
                    'max': float(cv_results[metric]['max'])
                } for metric in cv_results.keys()
            }
        },
        'bootstrap': {
            'n_bootstrap': 1000,
            'results': {
                metric: {
                    'mean': float(bootstrap_stats[metric]['mean']),
                    'std': float(bootstrap_stats[metric]['std']),
                    'ci_95_lower': float(bootstrap_stats[metric]['ci_95_lower']),
                    'ci_95_upper': float(bootstrap_stats[metric]['ci_95_upper']),
                    'ci_99_lower': float(bootstrap_stats[metric]['ci_99_lower']),
                    'ci_99_upper': float(bootstrap_stats[metric]['ci_99_upper'])
                } for metric in bootstrap_stats.keys()
            }
        },
        'learning_curves': {
            'final_train_score': float(learning_curves['train_mean'][-1]),
            'final_val_score': float(learning_curves['val_mean'][-1]),
            'overfitting_gap': float(learning_curves['train_mean'][-1] - learning_curves['val_mean'][-1])
        },
        'validation_curves': {
            'best_c_value': float(validation_curves['param_range'][np.argmax(validation_curves['val_mean'])]),
            'best_val_score': float(validation_curves['val_mean'].max())
        },
        'error_analysis': error_analysis['error_analysis'] if error_analysis['error_analysis'] else None
    }
    
    with open(results_dir / 'model_validation_summary.json', 'w') as f:
        json.dump(validation_summary, f, indent=2)
    
    logger.info(f"Risultati salvati in: {results_dir}")

def log_validation_to_mlflow(cv_results, bootstrap_stats, learning_curves, validation_curves, experiment_id):
    """
    Logga i risultati della validazione in MLflow.
    """
    logger.info("=== LOGGING VALIDAZIONE A MLFLOW ===")
    
    # Preparare parametri
    params = {
        'validation_type': 'comprehensive_model_validation',
        'cv_folds': CV_FOLDS,
        'n_bootstrap': 1000,
        'random_state': RANDOM_STATE
    }
    
    # Preparare metriche
    metrics = {
        'cv_recall_mean': cv_results['recall']['mean'],
        'cv_recall_std': cv_results['recall']['std'],
        'cv_f1_mean': cv_results['f1']['mean'],
        'cv_f1_std': cv_results['f1']['std'],
        'bootstrap_recall_mean': bootstrap_stats['recall']['mean'],
        'bootstrap_recall_ci_95_lower': bootstrap_stats['recall']['ci_95_lower'],
        'bootstrap_recall_ci_95_upper': bootstrap_stats['recall']['ci_95_upper'],
        'learning_curve_overfitting_gap': learning_curves['train_mean'][-1] - learning_curves['val_mean'][-1],
        'validation_curve_best_score': validation_curves['val_mean'].max()
    }
    
    # Logga in MLflow
    model_name = "model_validation"
    artifacts_path = PLOTS_DIR if PLOTS_DIR.exists() else None
    model_details = create_mlflow_run(model_name, params, metrics, artifacts_path, None)
    
    return model_details

def main():
    """
    Funzione principale per la validazione del modello.
    """
    logger.info("=== INIZIO MODEL VALIDATION ===")
    
    try:
        # 1. Setup MLflow
        experiment_id = setup_mlflow()
        
        # 2. Caricamento dati e modello
        X_train, X_test, y_train, y_test, model = load_data_and_model()
        
        # 3. Cross-validation
        cv_results = perform_cross_validation(model, X_train, y_train)
        
        # 4. Bootstrap validation
        bootstrap_results, bootstrap_stats = perform_bootstrap_validation(model, X_test, y_test)
        
        # 5. Learning curves
        learning_curves = analyze_learning_curves(model, X_train, y_train)
        
        # 6. Validation curves
        validation_curves = analyze_validation_curves(model, X_train, y_train)
        
        # 7. Error analysis
        error_analysis = analyze_error_analysis(model, X_test, y_test)
        
        # 8. Creazione plots
        create_validation_plots(cv_results, bootstrap_results, learning_curves, validation_curves, error_analysis, PLOTS_DIR)
        
        # 9. Salvare risultati
        save_validation_results(cv_results, bootstrap_stats, learning_curves, validation_curves, error_analysis, RESULTS_DIR)
        
        # 10. Logging a MLflow
        mlflow_details = log_validation_to_mlflow(cv_results, bootstrap_stats, learning_curves, validation_curves, experiment_id)
        
        # 11. Riepilogo finale
        logger.info(f"\n{'='*50}")
        logger.info("🏆 MODEL VALIDATION COMPLETATA")
        logger.info(f"{'='*50}")
        logger.info(f"📊 Cross-Validation Results:")
        logger.info(f"  Recall: {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
        logger.info(f"  F1-Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
        logger.info(f"  Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        
        logger.info(f"\n📊 Bootstrap Validation (95% CI):")
        logger.info(f"  Recall: {bootstrap_stats['recall']['mean']:.4f} [{bootstrap_stats['recall']['ci_95_lower']:.4f}, {bootstrap_stats['recall']['ci_95_upper']:.4f}]")
        logger.info(f"  F1-Score: {bootstrap_stats['f1']['mean']:.4f} [{bootstrap_stats['f1']['ci_95_lower']:.4f}, {bootstrap_stats['f1']['ci_95_upper']:.4f}]")
        
        logger.info(f"\n📈 Learning Curves:")
        logger.info(f"  Final Train Score: {learning_curves['train_mean'][-1]:.4f}")
        logger.info(f"  Final Val Score: {learning_curves['val_mean'][-1]:.4f}")
        logger.info(f"  Overfitting Gap: {learning_curves['train_mean'][-1] - learning_curves['val_mean'][-1]:.4f}")
        
        logger.info(f"\n🎯 Validation Curves:")
        logger.info(f"  Best C Value: {validation_curves['param_range'][np.argmax(validation_curves['val_mean'])]:.4f}")
        logger.info(f"  Best Val Score: {validation_curves['val_mean'].max():.4f}")
        
        if error_analysis['error_analysis']:
            logger.info(f"\n❌ Error Analysis:")
            logger.info(f"  Total Errors: {error_analysis['error_analysis']['error_count']}")
            logger.info(f"  False Positives: {error_analysis['error_analysis']['false_positives']}")
            logger.info(f"  False Negatives: {error_analysis['error_analysis']['false_negatives']}")
        
        logger.info(f"{'='*50}")
        
        return {
            'cv_results': cv_results,
            'bootstrap_results': bootstrap_results,
            'bootstrap_stats': bootstrap_stats,
            'learning_curves': learning_curves,
            'validation_curves': validation_curves,
            'error_analysis': error_analysis,
            'experiment_id': experiment_id
        }
        
    except Exception as e:
        logger.error(f"Errore durante model validation: {e}")
        raise

if __name__ == "__main__":
    main() 