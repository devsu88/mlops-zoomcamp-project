terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }

  backend "gcs" {
    bucket = "mlops-breast-cancer-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "storage.googleapis.com",
    "run.googleapis.com",

    "aiplatform.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com"
  ])

  service = each.value
  project = var.project_id
}

# Network module
module "network" {
  source = "./modules/network"

  project_id = var.project_id
  region     = var.region
  vpc_name   = "mlops-vpc"
}

# Storage module
module "storage" {
  source = "./modules/storage"

  project_id = var.project_id
  region     = var.region

  buckets = {
    "mlops-breast-cancer-data" = {
      location = "EUROPE-WEST1"
      labels = {
        environment = var.environment
        purpose     = "ml-data"
      }
    }
    "mlops-breast-cancer-models" = {
      location = "EUROPE-WEST1"
      labels = {
        environment = var.environment
        purpose     = "ml-models"
      }
    }
    "mlops-breast-cancer-artifacts" = {
      location = "EUROPE-WEST1"
      labels = {
        environment = var.environment
        purpose     = "ml-artifacts"
      }
    }
    "mlops-breast-cancer-monitoring" = {
      location = "EUROPE-WEST1"
      labels = {
        environment = var.environment
        purpose     = "ml-monitoring"
      }
    }
  }
}



# MLflow module - COMMENTATO PER DEPLOY BASE
module "mlflow" {
  source = "./modules/mlflow"

  project_id = var.project_id
  region     = var.region

  artifact_bucket = module.storage.bucket_names["mlops-breast-cancer-artifacts"]
}

# Prefect module - COMMENTATO PER DEPLOY BASE
module "prefect" {
  source = "./modules/prefect"

  project_id = var.project_id
  region     = var.region
}

# Monitoring module - COMMENTATO PER DEPLOY BASE
module "monitoring" {
  source = "./modules/monitoring"

  project_id = var.project_id
  region     = var.region

  data_bucket = module.storage.bucket_names["mlops-breast-cancer-data"]
  monitoring_bucket = module.storage.bucket_names["mlops-breast-cancer-monitoring"]
  models_bucket = module.storage.bucket_names["mlops-breast-cancer-models"]
}

# Deployment module - COMMENTATO PER DEPLOY BASE
module "deployment" {
  source = "./modules/deployment"

  project_id = var.project_id
  region     = var.region

  model_bucket = module.storage.bucket_names["mlops-breast-cancer-models"]
  mlflow_tracking_uri = module.mlflow.tracking_uri
}
