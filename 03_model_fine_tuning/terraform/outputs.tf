output "bucket_name" {
  value       = google_storage_bucket.pipeline_bucket.name
  description = "The GCS bucket created for pipeline artifacts."
}

output "artifact_registry_repo" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.container_repo.repository_id}"
  description = "Artifact Registry Docker repository URI."
}

output "pipeline_sa_email" {
  value       = google_service_account.pipeline_sa.email
  description = "The service account email running pipeline jobs."
}
