variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "mlops-breast-cancer"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "europe-west1-b"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "database_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "prefect_database_password" {
  description = "Prefect database password"
  type        = string
  sensitive   = true
}
