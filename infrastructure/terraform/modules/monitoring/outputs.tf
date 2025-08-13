output "evidently_url" {
  description = "The URL of the Evidently Dashboard"
  value       = google_cloud_run_service.evidently.status[0].url
}

output "batch_monitoring_url" {
  description = "The URL of the Batch Monitoring service"
  value       = google_cloud_run_service.batch_monitoring.status[0].url
}

output "evidently_service_name" {
  description = "The name of the Evidently Dashboard service"
  value       = google_cloud_run_service.evidently.name
}

output "batch_monitoring_service_name" {
  description = "The name of the Batch Monitoring service"
  value       = google_cloud_run_service.batch_monitoring.name
}
