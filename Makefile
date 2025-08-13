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

# =============================================================================
# DOCKER IMAGES BUILD
# =============================================================================

.PHONY: docker-build-mlflow docker-build-prefect docker-build-evidently docker-build-batch-monitoring docker-build-all

docker-build-mlflow:
	@echo "🐳 Building MLflow Docker image..."
	docker build -f Dockerfile.mlflow -t gcr.io/mlops-breast-cancer/mlflow:latest .
	@echo "✅ MLflow image built successfully"

docker-build-prefect:
	@echo "🐳 Building Prefect Docker image..."
	docker build -f Dockerfile.prefect -t gcr.io/mlops-breast-cancer/prefect:latest .
	@echo "✅ Prefect image built successfully"

docker-build-evidently:
	@echo "🐳 Building Evidently Docker image..."
	docker build -f Dockerfile.evidently -t gcr.io/mlops-breast-cancer/evidently:latest .
	@echo "✅ Evidently image built successfully"

docker-build-batch-monitoring:
	@echo "🐳 Building Batch Monitoring Docker image..."
	docker build -f Dockerfile.batch-monitoring -t gcr.io/mlops-breast-cancer/batch-monitoring:latest .
	@echo "✅ Batch Monitoring image built successfully"

docker-build-api:
	@echo "🐳 Building API Docker image..."
	docker build -f Dockerfile -t gcr.io/mlops-breast-cancer/breast-cancer-api:latest .
	@echo "✅ API image built successfully"

docker-build-all: docker-build-mlflow docker-build-prefect docker-build-evidently docker-build-batch-monitoring docker-build-api
	@echo "🎉 All Docker images built successfully!"

# =============================================================================
# DOCKER PUSH TO GCR
# =============================================================================

.PHONY: docker-push-mlflow docker-push-prefect docker-push-evidently docker-push-batch-monitoring docker-push-all

docker-push-mlflow:
	@echo "📤 Pushing MLflow image to GCR..."
	docker push gcr.io/mlops-breast-cancer/mlflow:latest
	@echo "✅ MLflow image pushed successfully"

docker-push-prefect:
	@echo "📤 Pushing Prefect image to GCR..."
	docker push gcr.io/mlops-breast-cancer/prefect:latest
	@echo "✅ Prefect image pushed successfully"

docker-push-evidently:
	@echo "📤 Pushing Evidently image to GCR..."
	docker push gcr.io/mlops-breast-cancer/evidently:latest
	@echo "✅ Evidently image pushed successfully"

docker-push-batch-monitoring:
	@echo "📤 Pushing Batch Monitoring image to GCR..."
	docker push gcr.io/mlops-breast-cancer/batch-monitoring:latest
	@echo "✅ Batch Monitoring image pushed successfully"

docker-push-api:
	@echo "📤 Pushing API image to GCR..."
	docker push gcr.io/mlops-breast-cancer/breast-cancer-api:latest
	@echo "✅ API image pushed successfully"

docker-push-all: docker-push-mlflow docker-push-prefect docker-push-evidently docker-push-batch-monitoring docker-push-api
	@echo "🎉 All Docker images pushed successfully!"

# =============================================================================
# TERRAFORM DEPLOYMENT
# =============================================================================

.PHONY: terraform-init terraform-plan terraform-apply terraform-destroy terraform-apply-api terraform-apply-mlflow terraform-apply-prefect terraform-apply-evidently

terraform-init:
	@echo "🔧 Initializing Terraform..."
	cd infrastructure/terraform && terraform init
	@echo "✅ Terraform initialized successfully"

terraform-plan:
	@echo "📋 Planning Terraform deployment..."
	cd infrastructure/terraform && terraform plan -out=tfplan
	@echo "✅ Terraform plan created successfully"

terraform-apply:
	@echo "🚀 Applying Terraform configuration..."
	cd infrastructure/terraform && terraform apply tfplan
	@echo "✅ Terraform deployment completed successfully"

terraform-destroy:
	@echo "🗑️  Destroying Terraform infrastructure..."
	cd infrastructure/terraform && terraform destroy -auto-approve
	@echo "✅ Terraform infrastructure destroyed successfully"

# Target specifici per moduli singoli
terraform-apply-api:
	@echo "🚀 Applying Terraform configuration for API only..."
	cd infrastructure/terraform && terraform apply -target=module.deployment
	@echo "✅ API Terraform deployment completed successfully"

terraform-apply-mlflow:
	@echo "🚀 Applying Terraform configuration for MLflow only..."
	cd infrastructure/terraform && terraform apply -target=module.mlflow
	@echo "✅ MLflow Terraform deployment completed successfully"

terraform-apply-prefect:
	@echo "🚀 Applying Terraform configuration for Prefect only..."
	cd infrastructure/terraform && terraform apply -target=module.prefect
	@echo "✅ Prefect Terraform deployment completed successfully"

terraform-apply-evidently:
	@echo "🚀 Applying Terraform configuration for Evidently only..."
	cd infrastructure/terraform && terraform apply -target=module.monitoring
	@echo "✅ Evidently Terraform deployment completed successfully"

# =============================================================================
# COMPLETE DEPLOYMENT PIPELINE
# =============================================================================

.PHONY: deploy-complete deploy-api-only deploy-mlflow-only deploy-prefect-only deploy-evidently-only

deploy-api-only: docker-build-api docker-push-api terraform-apply-api
	@echo "🎉 API deployment completed successfully!"
	@echo "📊 Check the API service:"
	@echo "   - API: https://breast-cancer-api-xxx-ew.a.run.app"

deploy-mlflow-only: docker-build-mlflow docker-push-mlflow terraform-apply-mlflow
	@echo "🎉 MLflow deployment completed successfully!"
	@echo "📊 Check the MLflow service:"
	@echo "   - MLflow: https://mlflow-server-xxx-ew.a.run.app"

deploy-prefect-only: docker-build-prefect docker-push-prefect terraform-apply-prefect
	@echo "🎉 Prefect deployment completed successfully!"
	@echo "📊 Check the Prefect service:"
	@echo "   - Prefect: https://prefect-server-xxx-ew.a.run.app"

deploy-evidently-only: docker-build-evidently docker-push-evidently docker-build-batch-monitoring docker-push-batch-monitoring terraform-apply-evidently
	@echo "🎉 Evidently deployment completed successfully!"
	@echo "📊 Check the Evidently services:"
	@echo "   - Evidently Dashboard: https://evidently-dashboard-xxx-ew.a.run.app"
	@echo "   - Batch Monitoring: https://batch-monitoring-xxx-ew.a.run.app"

deploy-complete: docker-build-all docker-push-all terraform-apply
	@echo "🎉 Complete MLOps infrastructure deployed successfully!"
	@echo "📊 Check the following services:"
	@echo "   - MLflow: https://mlflow-server-xxx-ew.a.run.app"
	@echo "   - Prefect: https://prefect-server-xxx-ew.a.run.app"
	@echo "   - Evidently Dashboard: https://evidently-dashboard-xxx-ew.a.run.app"
	@echo "   - Batch Monitoring: https://batch-monitoring-xxx-ew.a.run.app"
	@echo "   - API: https://breast-cancer-api-xxx-ew.a.run.app"

# =============================================================================
# GCP DEPLOYMENT
# =============================================================================

.PHONY: gcp-build gcp-push gcp-deploy gcp-deploy-all

gcp-build:
	@echo "🏗️  Building Docker image for GCP..."
	docker build -t gcr.io/mlops-breast-cancer/mlops-breast-cancer:latest .
	@echo "✅ Docker image built successfully"

gcp-push:
	@echo "📤 Pushing Docker image to GCP Container Registry..."
	docker push gcr.io/mlops-breast-cancer/mlops-breast-cancer:latest
	@echo "✅ Docker image pushed successfully"

gcp-deploy:
	@echo "🚀 Deploying to Cloud Run..."
	gcloud run deploy mlops-breast-cancer-api \
		--image gcr.io/mlops-breast-cancer/mlops-breast-cancer:latest \
		--platform managed \
		--region europe-west1 \
		--allow-unauthenticated \
		--port 8000 \
		--memory 2Gi \
		--cpu 1 \
		--max-instances 10 \
		--timeout 300
	@echo "✅ Deployment completed successfully"

gcp-deploy-all: gcp-build gcp-push gcp-deploy
	@echo "🎉 Complete GCP deployment pipeline executed successfully!"

gcp-status:
	@echo "📊 Checking Cloud Run service status..."
	gcloud run services describe mlops-breast-cancer-api --region=europe-west1 --format="value(status.url)"

gcp-logs:
	@echo "📋 Fetching Cloud Run logs..."
	gcloud logs read --service=mlops-breast-cancer-api --limit=50 --format="table(timestamp,severity,textPayload)"
