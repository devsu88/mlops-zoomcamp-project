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

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = module.database.connection_name
}

output "database_name" {
  description = "Database name"
  value       = module.database.database_name
}

output "database_user" {
  description = "Database user"
  value       = module.database.database_user
}
