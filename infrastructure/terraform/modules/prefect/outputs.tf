output "server_url" {
  description = "Prefect server URL"
  value       = google_cloud_run_service.prefect.status[0].url
}

output "service_name" {
  description = "Prefect service name"
  value       = google_cloud_run_service.prefect.name
}
