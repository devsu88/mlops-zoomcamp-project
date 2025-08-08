.PHONY: help install test lint format clean deploy setup-dev download-data train serve test-api monitoring-setup data-quality-check drift-detection performance-monitoring evidently-dashboard monitoring-all

help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting"
	@echo "  format        - Format code"
	@echo "  clean         - Clean up"
	@echo "  deploy        - Deploy to GCP"
	@echo "  setup-dev     - Setup development environment"
	@echo "  download-data - Download breast cancer dataset"
	@echo "  train         - Train model"
	@echo "  serve         - Serve model locally"
	@echo "  test-api      - Test API endpoints"
	@echo ""
	@echo "Monitoring commands:"
	@echo "  monitoring-setup      - Setup monitoring directories"
	@echo "  data-quality-check    - Check data quality"
	@echo "  drift-detection       - Detect data drift"
	@echo "  performance-monitoring - Monitor model performance"
	@echo "  evidently-dashboard   - Start Evidently dashboard"
	@echo "  monitoring-all        - Run all monitoring checks"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-api:
	pytest tests/test_api.py -v

test-monitoring:
	pytest tests/test_monitoring.py -v

lint:
	black --check --diff src/ tests/
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports --disallow-untyped-defs

format:
	black src/ tests/
	isort src/ tests/

security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

ci: lint test-cov security

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

deploy:
	cd infrastructure/terraform && terraform apply -var-file="environments/dev.tfvars"

setup-dev:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	pre-commit install

download-data:
	python src/data/download_dataset.py

preprocess:
	python src/data/preprocessing.py

train:
	python src/models/train_model.py

train-baseline:
	python src/models/train_baseline.py

train-mlflow:
	python src/models/train_with_mlflow.py

tune-hyperparameters:
	python src/models/hyperparameter_tuning.py

analyze-features:
	python src/models/feature_importance_analysis.py

validate-model:
	python src/models/model_validation.py

# Prefect Orchestration
setup-prefect:
	python src/orchestration/prefect_setup.py

start-workers:
	python src/orchestration/start_workers.py

run-pipeline:
	python src/orchestration/prefect_workflows.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Monitoring commands
monitoring-setup:
	PYTHONPATH=. python src/monitoring/monitoring_config.py

data-quality-check:
	PYTHONPATH=. python src/monitoring/data_quality.py

drift-detection:
	PYTHONPATH=. python src/monitoring/drift_detection.py

performance-monitoring:
	PYTHONPATH=. python src/monitoring/performance_monitoring.py

evidently-dashboard:
	PYTHONPATH=. python src/monitoring/evidently_dashboard.py

monitoring-all: monitoring-setup data-quality-check drift-detection performance-monitoring evidently-dashboard
