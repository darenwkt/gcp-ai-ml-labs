output "bucket_name" {
  value       = google_storage_bucket.pipeline_bucket.name
  description = "The name of the GCS bucket for pipeline artifacts"
}

output "artifact_registry_repo" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}"
  description = "The URI of the Artifact Registry repository"
}

output "pipeline_sa_email" {
  value       = google_service_account.pipeline_sa.email
  description = "The email of the pipeline execution Service Account"
}
