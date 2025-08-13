variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "mlflow_user" {
  description = "MLflow database user"
  type        = string
}

variable "mlflow_password" {
  description = "MLflow database password"
  type        = string
  sensitive   = true
}

variable "prefect_user" {
  description = "Prefect database user"
  type        = string
}

variable "prefect_password" {
  description = "Prefect database password"
  type        = string
  sensitive   = true
}
