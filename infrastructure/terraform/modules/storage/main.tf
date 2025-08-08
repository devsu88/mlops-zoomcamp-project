resource "google_storage_bucket" "buckets" {
  for_each = var.buckets
  
  name          = each.key
  location      = each.value.location
  force_destroy = true
  
  labels = each.value.labels
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket_iam_member" "public_read" {
  for_each = var.buckets
  
  bucket = google_storage_bucket.buckets[each.key].name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
} 