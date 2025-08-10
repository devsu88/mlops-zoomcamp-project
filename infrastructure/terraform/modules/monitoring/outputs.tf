output "dashboard_url" {
  description = "Evidently dashboard URL"
  value       = google_cloud_run_service.evidently.status[0].url
}

output "function_url" {
  description = "Batch monitoring function URL"
  value       = google_cloudfunctions_function.batch_monitoring.https_trigger_url
}
