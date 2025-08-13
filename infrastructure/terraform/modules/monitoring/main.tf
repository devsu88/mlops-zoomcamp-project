# Cloud Run service per Evidently Dashboard
resource "google_cloud_run_service" "evidently" {
  name     = "evidently-dashboard"
  location = var.region

  template {
    spec {
      containers {
        image = var.evidently_image

        env {
          name  = "DATA_BUCKET"
          value = var.data_bucket
        }

        env {
          name  = "MONITORING_BUCKET"
          value = var.monitoring_bucket
        }

        env {
          name  = "ENVIRONMENT"
          value = "cloud"
        }

        ports {
          container_port = 8080
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Cloud Run service per Batch Monitoring
resource "google_cloud_run_service" "batch_monitoring" {
  name     = "batch-monitoring"
  location = var.region

  template {
    spec {
      containers {
        image = var.batch_monitoring_image

        env {
          name  = "DATA_BUCKET"
          value = var.data_bucket
        }

        env {
          name  = "MONITORING_BUCKET"
          value = var.monitoring_bucket
        }

        env {
          name  = "MODELS_BUCKET"
          value = var.models_bucket
        }

        env {
          name  = "ENVIRONMENT"
          value = "cloud"
        }

        ports {
          container_port = 8080
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Accesso pubblico per Evidently Dashboard
resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_service.evidently.location
  service  = google_cloud_run_service.evidently.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Accesso pubblico per Batch Monitoring
resource "google_cloud_run_service_iam_member" "batch_monitoring_public_access" {
  location = google_cloud_run_service.batch_monitoring.location
  service  = google_cloud_run_service.batch_monitoring.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
