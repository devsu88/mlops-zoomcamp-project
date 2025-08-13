resource "google_cloud_run_service" "mlflow" {
  name     = "mlflow-server"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/mlflow:latest"

        command = ["mlflow"]
        args    = ["server", "--host", "0.0.0.0", "--port", "5000"]

        env {
          name  = "MLFLOW_TRACKING_URI"
          value = "postgresql://${var.database_user}:${var.database_password}@${var.database_connection_name}/${var.database_name}"
        }

        env {
          name  = "MLFLOW_ARTIFACT_ROOT"
          value = "gs://${var.artifact_bucket}"
        }

        env {
          name  = "MLFLOW_SERVE_ARTIFACTS"
          value = "true"
        }

        env {
          name  = "ENVIRONMENT"
          value = "cloud"
        }

        ports {
          container_port = 5000
          name           = "http1"
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "1Gi"
          }
        }
      }

      timeout_seconds = 300
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_service.mlflow.location
  service  = google_cloud_run_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
