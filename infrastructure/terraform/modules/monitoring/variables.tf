variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "data_bucket" {
  description = "Cloud Storage bucket for data"
  type        = string
}

variable "monitoring_bucket" {
  description = "Cloud Storage bucket for monitoring"
  type        = string
}
