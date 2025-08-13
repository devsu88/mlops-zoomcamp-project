resource "google_sql_database_instance" "instance" {
  name             = "mlops-database"
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-f1-micro"  # Free tier

    backup_configuration {
      enabled = true
    }

    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"
      }
    }
  }

  deletion_protection = false
}

# Database MLflow
resource "google_sql_database" "mlflow_database" {
  name     = "mlflow"
  instance = google_sql_database_instance.instance.name
}

# Database Prefect
resource "google_sql_database" "prefect_database" {
  name     = "prefect"
  instance = google_sql_database_instance.instance.name
}

# User MLflow
resource "google_sql_user" "mlflow_user" {
  name     = var.mlflow_user
  instance = google_sql_database_instance.instance.name
  password = var.mlflow_password
}

# User Prefect
resource "google_sql_user" "prefect_user" {
  name     = var.prefect_user
  instance = google_sql_database_instance.instance.name
  password = var.prefect_password
}
