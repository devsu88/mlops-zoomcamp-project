# MLflow URL - COMMENTATO PER DEPLOY BASE
# output "mlflow_url" {
#   description = "MLflow tracking server URL"
#   value       = module.mlflow.tracking_uri
# }

# Prefect URL - COMMENTATO PER DEPLOY BASE
# output "prefect_url" {
#   description = "Prefect server URL"
#   value       = module.prefect.server_url
# }

# API URL - COMMENTATO PER DEPLOY BASE
# output "api_url" {
#   description = "FastAPI deployment URL"
#   value       = module.deployment.api_url
# }

# Monitoring Dashboard URL - COMMENTATO PER DEPLOY BASE
# output "monitoring_dashboard_url" {
#   description = "Evidently monitoring dashboard URL"
#   value       = module.monitoring.dashboard_url
# }

output "storage_buckets" {
  description = "Created storage buckets"
  value       = module.storage.bucket_names
}

output "pipeline_api_url" {
  description = "MLOps Pipeline API URL (integrata in Prefect)"
  value       = module.prefect.api_url
}
