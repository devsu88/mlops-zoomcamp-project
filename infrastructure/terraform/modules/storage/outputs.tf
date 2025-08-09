output "bucket_names" {
  description = "Names of created buckets"
  value = {
    for k, v in google_storage_bucket.buckets : k => v.name
  }
} 