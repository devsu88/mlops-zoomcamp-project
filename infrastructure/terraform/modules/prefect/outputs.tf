output "server_url" {
  description = "Prefect server URL"
  value       = google_cloud_run_service.prefect.status[0].url
}

output "api_url" {
  description = "Pipeline API URL (porta 8000)"
  value       = "${google_cloud_run_service.prefect.status[0].url}:8000"
}

output "service_name" {
  description = "Prefect service name"
  value       = google_cloud_run_service.prefect.name
}
