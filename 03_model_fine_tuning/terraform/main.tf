provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. GCS Bucket for Pipeline Artifacts
resource "google_storage_bucket" "pipeline_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# 2. Artifact Registry for Container Images
resource "google_artifact_registry_repository" "container_repo" {
  location      = var.region
  repository_id = var.repository_id
  description   = "Docker repository for GPT2 IT Support Fine-Tuning Lab 03 images"
  format        = "DOCKER"
}

# 3. Custom Service Account for Pipeline Execution
resource "google_service_account" "pipeline_sa" {
  account_id   = var.pipeline_sa_name
  display_name = "Fine-Tuning Pipeline Custom Service Account"
}

# 4. IAM Bindings for Service Account
resource "google_project_iam_member" "sa_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

resource "google_project_iam_member" "sa_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

resource "google_project_iam_member" "sa_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

resource "google_project_iam_member" "sa_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

resource "google_project_iam_member" "sa_user_access" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}
