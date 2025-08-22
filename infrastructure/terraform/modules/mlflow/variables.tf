variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}



variable "artifact_bucket" {
  description = "Cloud Storage bucket for artifacts"
  type        = string
}
