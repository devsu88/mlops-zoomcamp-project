variable "project_id" {
  description = "The ID of the project"
  type        = string
}

variable "region" {
  description = "The region to deploy to"
  type        = string
}

variable "data_bucket" {
  description = "The name of the data bucket"
  type        = string
}

variable "monitoring_bucket" {
  description = "The name of the monitoring bucket"
  type        = string
}

variable "models_bucket" {
  description = "The name of the models bucket"
  type        = string
}

variable "evidently_image" {
  description = "The Docker image for Evidently Dashboard"
  type        = string
  default     = "gcr.io/mlops-breast-cancer/evidently:latest"
}

variable "batch_monitoring_image" {
  description = "The Docker image for Batch Monitoring"
  type        = string
  default     = "gcr.io/mlops-breast-cancer/batch-monitoring:latest"
}
