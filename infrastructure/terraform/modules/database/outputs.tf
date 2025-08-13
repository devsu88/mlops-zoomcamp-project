output "connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.instance.connection_name
}

output "mlflow_database_name" {
  description = "MLflow database name"
  value       = google_sql_database.mlflow_database.name
}

output "mlflow_user" {
  description = "MLflow database user"
  value       = google_sql_user.mlflow_user.name
}

output "prefect_database_name" {
  description = "Prefect database name"
  value       = google_sql_database.prefect_database.name
}

output "prefect_user" {
  description = "Prefect database user"
  value       = google_sql_user.prefect_user.name
}
