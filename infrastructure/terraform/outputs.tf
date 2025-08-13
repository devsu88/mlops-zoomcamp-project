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

output "mlflow_database_name" {
  description = "MLflow database name"
  value       = module.database.mlflow_database_name
}

output "mlflow_user" {
  description = "MLflow database user"
  value       = module.database.mlflow_user
}

output "prefect_database_name" {
  description = "Prefect database name"
  value       = module.database.prefect_database_name
}

output "prefect_user" {
  description = "Prefect database user"
  value       = module.database.prefect_user
}
