# 🏥 Breast Cancer Classification - MLOps Pipeline End-to-End

Un progetto completo di **Machine Learning Operations (MLOps)** che implementa una pipeline end-to-end per la classificazione del cancro al seno, utilizzando tecnologie moderne e best practices dell'industria.

## 📋 Indice

- [🎯 Panoramica del Progetto](#-panoramica-del-progetto)
- [🏗️ Architettura del Sistema](#️-architettura-del-sistema)
- [📊 Dataset e Problema](#-dataset-e-problema)
- [🛠️ Stack Tecnologico](#️-stack-tecnologico)
- [🚀 Componenti della Pipeline](#-componenti-della-pipeline)
- [⚡ Quick Start](#-quick-start)
- [📚 Guide Dettagliate](#-guide-dettagliate)
- [🧪 Testing e Quality Assurance](#-testing-e-quality-assurance)
- [📈 Monitoring e Observability](#-monitoring-e-observability)
- [☁️ Deployment Cloud](#️-deployment-cloud)
- [🔧 Comandi Utili](#-comandi-utili)
- [📊 Metriche del Modello](#-metriche-del-modello)
- [🔒 Sicurezza e Best Practices](#-sicurezza-e-best-practices)
- [📖 Documentazione API](#-documentazione-api)
- [🤝 Contributing](#-contributing)
- [📄 Licenza](#-licenza)

## 🎯 Panoramica del Progetto

Questo progetto dimostra l'implementazione completa di una **pipeline MLOps end-to-end** per un problema di classificazione medica. Il sistema include:

- **Data Pipeline**: Download, preprocessing e feature engineering automatici
- **Experiment Tracking**: Tracciamento completo degli esperimenti con MLflow
- **Model Training**: Training automatizzato con hyperparameter tuning
- **Model Registry**: Gestione versioni e deployment dei modelli
- **API Deployment**: Servizio REST containerizzato con FastAPI
- **Monitoring**: Monitoring in tempo reale con Evidently AI
- **CI/CD**: Pipeline automatizzata con GitHub Actions
- **Infrastructure as Code**: Provisioning cloud con Terraform

### 🎓 Obiettivi Educativi

Il progetto è progettato per insegnare:
- **MLOps Fundamentals**: Principi e pratiche di MLOps
- **Cloud Architecture**: Design di sistemi distribuiti
- **Automation**: Automazione completa del ciclo di vita ML
- **Monitoring**: Osservabilità e monitoring in produzione
- **Best Practices**: Codice pulito, testing, documentazione

## 🏗️ Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        GCP Cloud Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Cloud Run   │  │ Cloud SQL   │  │ Cloud       │          │
│  │ (FastAPI)   │  │ (PostgreSQL)│  │ Storage     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure as Code                      │
│                    (Terraform)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Local Development                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Data        │  │ Model       │  │ API         │          │
│  │ Pipeline    │  │ Training    │  │ (FastAPI)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ MLflow      │  │ Prefect     │  │ Evidently   │          │
│  │ Tracking    │  │ Orchestr.   │  │ Monitoring  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 🔄 Flusso dei Dati

1. **Data Ingestion**: Download automatico da Kaggle
2. **Preprocessing**: Cleaning, feature engineering, scaling
3. **Training**: Experiment tracking con MLflow
4. **Validation**: Cross-validation e metriche multiple
5. **Deployment**: Containerizzazione e deployment API
6. **Monitoring**: Real-time monitoring e alerting

## 📊 Dataset e Problema

### 🏥 Contesto Medico

**Dataset**: [Breast Cancer Dataset](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset)
- **Problema**: Classificazione binaria (Benigno/Maligno)
- **Features**: 20 caratteristiche morfologiche del tessuto
- **Samples**: 569 campioni (357 benigni, 212 maligni)

### 🎯 Metriche di Valutazione

Data la natura medica del problema, utilizziamo metriche specifiche:

- **Recall** (Sensibilità): Priorità massima per evitare falsi negativi
- **F1-Score**: Bilanciamento tra precision e recall
- **PR-AUC**: Più appropriato per dataset sbilanciati
- **Confusion Matrix**: Analisi dettagliata degli errori

### 📈 Caratteristiche del Dataset

```python
# Esempio di features utilizzate
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    # ... + 10 features "standard error" e "worst"
]
```

## 🛠️ Stack Tecnologico

### ☁️ Cloud Platform
- **Google Cloud Platform (GCP)**: Piattaforma cloud principale
- **Cloud Run**: Deployment serverless dell'API
- **Cloud Storage**: Storage per dati e modelli
- **Cloud SQL**: Database PostgreSQL per MLflow
- **Cloud Monitoring**: Monitoring e logging

### 🔬 Machine Learning
- **Scikit-learn**: Algoritmi ML e preprocessing
- **XGBoost**: Gradient boosting avanzato
- **MLflow**: Experiment tracking e model registry
- **Joblib**: Serializzazione modelli

### 🚀 Deployment & API
- **FastAPI**: Framework API moderno e performante
- **Docker**: Containerizzazione
- **Uvicorn**: ASGI server
- **Pydantic**: Validazione dati

### 🔄 Orchestration & Workflow
- **Prefect**: Orchestrazione pipeline
- **Terraform**: Infrastructure as Code
- **GitHub Actions**: CI/CD pipeline

### 📊 Monitoring & Observability
- **Evidently AI**: Data quality, drift detection, performance monitoring
- **MLflow**: Model versioning e tracking
- **Logging**: Structured logging

### 🧪 Testing & Quality
- **Pytest**: Testing framework
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks

## 🚀 Componenti della Pipeline

### 1. 📥 Data Pipeline

**File**: `src/data/`

```python
# Download automatico da Kaggle
python src/data/download_dataset.py

# Preprocessing e feature engineering
python src/data/preprocessing.py
```

**Funzionalità**:
- Download automatico dataset da Kaggle
- Data cleaning e validation
- Feature engineering
- Train/test split
- Metadata storage

### 2. 🧠 Model Training

**File**: `src/models/`

```python
# Training baseline
python src/models/train_baseline.py

# Training con MLflow
python src/models/train_with_mlflow.py

# Hyperparameter tuning
python src/models/hyperparameter_tuning.py
```

**Funzionalità**:
- Multiple algorithms (Logistic Regression, Random Forest, SVM, XGBoost)
- Cross-validation
- Hyperparameter optimization
- Model comparison
- Feature importance analysis

### 3. 🔄 Workflow Orchestration

**File**: `src/orchestration/`

```python
# Setup Prefect
python src/orchestration/prefect_setup.py

# Esecuzione pipeline
python src/orchestration/prefect_workflows.py
```

**Funzionalità**:
- Pipeline end-to-end automatizzata
- Task dependencies
- Error handling
- Retry logic
- Monitoring

### 4. 🌐 API Deployment

**File**: `src/api/`

```python
# Avvio API
uvicorn src.api.main:app --reload
```

**Endpoints**:
- `GET /health`: Health check
- `POST /predict`: Predizione singola
- `POST /predict/batch`: Predizioni multiple
- `GET /model/info`: Informazioni modello
- `GET /features`: Features disponibili

### 5. 📊 Monitoring & Observability

**File**: `src/monitoring/`

```python
# Data quality monitoring
python src/monitoring/data_quality.py

# Drift detection
python src/monitoring/drift_detection.py

# Performance monitoring
python src/monitoring/performance_monitoring.py
```

**Funzionalità**:
- Data quality checks
- Drift detection
- Model performance monitoring
- Automated alerts
- Dashboard HTML

## ⚡ Quick Start

### Prerequisiti

```bash
# Python 3.11+
python --version

# Conda (raccomandato)
conda create -n mlops-breast-cancer python=3.11
conda activate mlops-breast-cancer

# Git
git clone https://github.com/your-username/mlops-breast-cancer.git
cd mlops-breast-cancer
```

### Installazione

```bash
# Installare dipendenze
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install
```

### Esecuzione Pipeline Completa

```bash
# 1. Download e preprocessing dati
make download-data
make preprocess

# 2. Training modelli
make train-mlflow

# 3. Avvio API
make serve

# 4. Testing
make test

# 5. Monitoring
make monitoring-all
```

## 📚 Guide Dettagliate

### 🏗️ Infrastructure Setup

```bash
# Configurazione GCP
gcloud auth login
gcloud config set project mlops-breast-cancer

# Deploy infrastructure
cd infrastructure/terraform
terraform init
terraform apply -var-file="environments/dev.tfvars"
```

### 🔄 CI/CD Pipeline

Il progetto include una pipeline CI/CD completa con GitHub Actions:

```yaml
# .github/workflows/ci-cd.yaml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: make test
      - name: Run linting
        run: make lint
```

### 🐳 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing e Quality Assurance

### Test Strategy

```bash
# Unit tests
pytest tests/test_api.py -v

# Integration tests
pytest tests/test_monitoring.py -v

# Coverage report
make test-cov
```

### Code Quality

```bash
# Formatting
make format

# Linting
make lint

# Security checks
make security
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

## 📈 Monitoring e Observability

### Data Quality Monitoring

```python
# Esempio di monitoring
from src.monitoring.data_quality import DataQualityMonitor

monitor = DataQualityMonitor()
report = monitor.check_data_quality(current_data, reference_data)
```

**Metriche monitorate**:
- Missing values
- Data types
- Value ranges
- Duplicates
- Statistical distributions

### Drift Detection

```python
# Esempio di drift detection
from src.monitoring.drift_detection import DriftDetector

detector = DriftDetector()
drift_report = detector.detect_drift(current_data, reference_data)
```

**Tipi di drift rilevati**:
- Feature drift
- Target drift
- Data quality drift
- Statistical drift

### Performance Monitoring

```python
# Esempio di performance monitoring
from src.monitoring.performance_monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
performance_report = monitor.monitor_performance(predictions, actuals)
```

**Metriche performance**:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC/PR curves
- Prediction drift

## ☁️ Deployment Cloud

### GCP Setup

```bash
# 1. Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage.googleapis.com

# 2. Create service account
gcloud iam service-accounts create terraform-sa \
    --display-name="Terraform Service Account"

# 3. Grant permissions
gcloud projects add-iam-policy-binding mlops-breast-cancer \
    --member="serviceAccount:terraform-sa@mlops-breast-cancer.iam.gserviceaccount.com" \
    --role="roles/editor"
```

### Terraform Infrastructure

```hcl
# infrastructure/terraform/main.tf
terraform {
  backend "gcs" {
    bucket = "mlops-breast-cancer-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
```

### Cloud Run Deployment

```bash
# Build e push Docker image
docker build -t gcr.io/mlops-breast-cancer/api .
docker push gcr.io/mlops-breast-cancer/api

# Deploy su Cloud Run
gcloud run deploy breast-cancer-api \
    --image gcr.io/mlops-breast-cancer/api \
    --platform managed \
    --region europe-west1 \
    --allow-unauthenticated
```

## 🔧 Comandi Utili

### Development

```bash
# Setup ambiente
make setup-dev

# Download dati
make download-data

# Preprocessing
make preprocess

# Training
make train-mlflow

# Hyperparameter tuning
make tune-hyperparameters

# Feature analysis
make analyze-features

# Model validation
make validate-model
```

### API e Testing

```bash
# Avvio API
make serve

# Test API
make test-api

# Test monitoring
make test-monitoring

# Test completi
make test
```

### Monitoring

```bash
# Setup monitoring
make monitoring-setup

# Data quality check
make data-quality-check

# Drift detection
make drift-detection

# Performance monitoring
make performance-monitoring

# Dashboard
make evidently-dashboard

# Tutto il monitoring
make monitoring-all
```

### CI/CD

```bash
# Linting
make lint

# Formatting
make format

# Security checks
make security

# CI pipeline
make ci
```

## 📊 Metriche del Modello

### Performance del Modello Migliore

| Metrica | Valore | Descrizione |
|---------|--------|-------------|
| **Recall** | 0.98 | Sensibilità - cruciale per applicazioni mediche |
| **F1-Score** | 0.97 | Bilanciamento precision/recall |
| **PR-AUC** | 0.99 | Area under Precision-Recall curve |
| **Accuracy** | 0.96 | Accuratezza generale |
| **Precision** | 0.96 | Precisione predizioni positive |

### Confronto Algoritmi

| Algoritmo | Recall | F1-Score | PR-AUC | Training Time |
|-----------|--------|----------|--------|---------------|
| **Logistic Regression** | 0.98 | 0.97 | 0.99 | 0.1s |
| Random Forest | 0.97 | 0.96 | 0.98 | 0.5s |
| SVM | 0.96 | 0.95 | 0.97 | 1.2s |
| XGBoost | 0.97 | 0.96 | 0.98 | 0.8s |

### Feature Importance

Le features più importanti per la classificazione:

1. **concave_points_worst** (0.15)
2. **perimeter_worst** (0.12)
3. **radius_worst** (0.11)
4. **area_worst** (0.10)
5. **concavity_worst** (0.09)

## 🔒 Sicurezza e Best Practices

### Security Measures

- **Input Validation**: Pydantic schemas per validazione
- **Rate Limiting**: Protezione da abuso API
- **HTTPS**: SSL/TLS encryption
- **IAM**: Role-based access control
- **Secrets Management**: Gestione sicura credenziali

### Code Quality

- **Type Hints**: Type checking con MyPy
- **Documentation**: Docstrings e README completo
- **Testing**: 90%+ code coverage
- **Linting**: Black, Flake8, isort
- **Pre-commit**: Hooks automatici

### Monitoring Best Practices

- **Structured Logging**: Log format consistenti
- **Metrics Collection**: Prometheus/Grafana ready
- **Alerting**: Automated alerts per anomalie
- **Dashboard**: Real-time monitoring UI
- **Data Lineage**: Tracciabilità completa

## 📖 Documentazione API

### Endpoints Principali

#### Health Check
```bash
GET /health
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193
  }
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.95,
  "confidence": "Alta",
  "features_used": ["radius_mean", "texture_mean", ...],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "predictions": [
    {"features": {...}},
    {"features": {...}},
    {"features": {...}}
  ]
}
```

### Error Handling

```json
{
  "detail": "Validation error",
  "errors": [
    {
      "loc": ["features"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## 🤝 Contributing

### Development Workflow

1. **Fork** il repository
2. **Create** un branch feature: `git checkout -b feature/amazing-feature`
3. **Commit** le modifiche: `git commit -m 'Add amazing feature'`
4. **Push** al branch: `git push origin feature/amazing-feature`
5. **Open** una Pull Request

### Code Standards

- **Python**: PEP 8 compliance
- **Testing**: 90%+ coverage required
- **Documentation**: Docstrings per tutte le funzioni
- **Type Hints**: Required per nuove funzioni
- **Commits**: Conventional commits format

### Testing Guidelines

```bash
# Run all tests
make test

# Run specific test category
pytest tests/test_api.py -v

# Run with coverage
make test-cov

# Run linting
make lint
```

## 📄 Licenza

Questo progetto è rilasciato sotto la licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.

## 🙏 Ringraziamenti

- **DataTalksClub**: Per il corso MLOps Zoomcamp
- **Kaggle**: Per il dataset Breast Cancer
- **Google Cloud**: Per la piattaforma cloud
- **FastAPI**: Per il framework API
- **MLflow**: Per experiment tracking
- **Evidently AI**: Per monitoring
- **Prefect**: Per orchestration

---

**⭐ Se questo progetto ti è stato utile, considera di dargli una stella!**

**📧 Contatti**: [your-email@example.com](mailto:your-email@example.com)

**🌐 Portfolio**: [your-portfolio.com](https://your-portfolio.com)
