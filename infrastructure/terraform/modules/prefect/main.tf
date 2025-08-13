resource "google_cloud_run_service" "prefect" {
  name     = "prefect-server"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/prefect:latest"

        env {
          name  = "PREFECT_API_URL"
          value = "http://0.0.0.0:4200"
        }

        env {
          name  = "PREFECT_SERVER_DATABASE_CONNECTION_URL"
          value = "postgresql://${var.database_user}:${var.database_password}@${var.database_connection_name}/${var.database_name}"
        }

        env {
          name  = "ENVIRONMENT"
          value = "cloud"
        }

        ports {
          container_port = 4200
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_service.prefect.location
  service  = google_cloud_run_service.prefect.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
