resource "google_cloud_run_service" "evidently" {
  name     = "evidently-dashboard"
  location = var.region
  
  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/evidently:latest"
        
        env {
          name  = "DATA_BUCKET"
          value = var.data_bucket
        }
        
        env {
          name  = "MONITORING_BUCKET"
          value = var.monitoring_bucket
        }
        
        ports {
          container_port = 8080
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
  location = google_cloud_run_service.evidently.location
  service  = google_cloud_run_service.evidently.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Function per batch monitoring
resource "google_cloudfunctions_function" "batch_monitoring" {
  name        = "batch-monitoring"
  description = "Batch monitoring function for Evidently"
  runtime     = "python39"
  
  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.function_bucket.name
  source_archive_object = google_storage_bucket_object.function_zip.name
  trigger_http         = true
  entry_point          = "run_monitoring"
}

resource "google_storage_bucket" "function_bucket" {
  name     = "${var.project_id}-function-bucket"
  location = var.region
}

resource "google_storage_bucket_object" "function_zip" {
  name   = "batch_monitoring.zip"
  bucket = google_storage_bucket.function_bucket.name
  source = "batch_monitoring.py"
} 