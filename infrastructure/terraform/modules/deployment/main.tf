resource "google_cloud_run_service" "api" {
  name     = "breast-cancer-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/breast-cancer-api:latest"

        env {
          name  = "MLFLOW_TRACKING_URI"
          value = var.mlflow_tracking_uri
        }

        env {
          name  = "MODEL_BUCKET"
          value = var.model_bucket
        }

        ports {
          container_port = 8080
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
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

resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_service.api.location
  service  = google_cloud_run_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Load balancer per l'API
resource "google_compute_global_forwarding_rule" "api_lb" {
  name       = "api-load-balancer"
  target     = google_compute_target_https_proxy.api_proxy.id
  port_range = "443"
}

resource "google_compute_target_https_proxy" "api_proxy" {
  name             = "api-https-proxy"
  url_map          = google_compute_url_map.api_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.api_cert.id]
}

resource "google_compute_url_map" "api_url_map" {
  name            = "api-url-map"
  default_service = google_compute_backend_service.api_backend.id
}

resource "google_compute_backend_service" "api_backend" {
  name        = "api-backend"
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 30

  backend {
    group = google_cloud_run_service.api.status[0].url
  }
}

resource "google_compute_managed_ssl_certificate" "api_cert" {
  name = "api-ssl-cert"

  managed {
    domains = ["api.mlops-breast-cancer.com"]
  }
}
