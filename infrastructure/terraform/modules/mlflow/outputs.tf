output "tracking_uri" {
  description = "MLflow tracking server URL"
  value       = google_cloud_run_service.mlflow.status[0].url
}

output "service_name" {
  description = "MLflow service name"
  value       = google_cloud_run_service.mlflow.name
}
