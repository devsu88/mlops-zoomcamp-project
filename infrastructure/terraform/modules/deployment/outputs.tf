output "api_url" {
  description = "FastAPI deployment URL"
  value       = google_cloud_run_service.api.status[0].url
}

# output "load_balancer_ip" {
#   description = "Load balancer IP address"
#   value       = google_compute_global_forwarding_rule.api_lb.ip_address
# }
