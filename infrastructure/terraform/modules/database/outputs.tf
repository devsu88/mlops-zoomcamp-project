output "connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.instance.connection_name
}

output "database_name" {
  description = "Database name"
  value       = google_sql_database.database.name
}

output "database_user" {
  description = "Database user"
  value       = google_sql_user.user.name
} 