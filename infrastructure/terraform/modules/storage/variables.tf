variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "buckets" {
  description = "Map of bucket configurations"
  type = map(object({
    location = string
    labels   = map(string)
  }))
}
