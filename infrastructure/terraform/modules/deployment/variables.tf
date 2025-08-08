variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  type        = string
}

variable "model_bucket" {
  description = "Cloud Storage bucket for models"
  type        = string
} 