resource "google_cloud_run_service" "mlflow" {
  name     = "mlflow-server"
  location = var.region
  
  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/mlflow:latest"
        
        env {
          name  = "MLFLOW_TRACKING_URI"
          value = "postgresql://${var.database_user}:${var.database_password}@${var.database_connection_name}/${var.database_name}"
        }
        
        env {
          name  = "MLFLOW_ARTIFACT_ROOT"
          value = "gs://${var.artifact_bucket}"
        }
        
        ports {
          container_port = 5000
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
  location = google_cloud_run_service.mlflow.location
  service  = google_cloud_run_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "allUsers"
} 