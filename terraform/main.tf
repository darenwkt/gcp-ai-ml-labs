terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.60.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 4.60.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.3.0"
    }
    null = {
      source  = "hashicorp/null"
      version = ">= 3.2.0"
    }
  }
}


provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}


# --- Project Services / APIs & Labels ---
locals {
  apis = [
    "aiplatform.googleapis.com",
    "cloudfunctions.googleapis.com",
    "pubsub.googleapis.com",
    "logging.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "eventarc.googleapis.com",
    "iam.googleapis.com"
  ]

  common_labels = {
    managed_by  = "terraform"
    project     = "anomaly-detection-pipeline"
    environment = "sandbox"
  }

  endpoint_id = "184013892"
}

resource "google_project_service" "services" {
  for_each                   = toset(local.apis)
  project                    = var.project_id
  service                    = each.key
  disable_dependent_services = false
  disable_on_destroy         = false
}

# --- Storage Buckets ---
resource "google_storage_bucket" "pipeline_bucket" {
  name                        = "${var.project_id}-anomaly-pipeline-artifacts"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
  
  labels = local.common_labels

  depends_on = [google_project_service.services]
}

resource "google_storage_bucket" "trigger_bucket" {
  name                        = "${var.project_id}-anomaly-trigger-source"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
  
  labels = local.common_labels

  depends_on = [google_project_service.services]
}

# --- Upload Datasets & Pipelines ---
resource "google_storage_bucket_object" "training_data" {
  name   = "data/training_data.csv"
  bucket = google_storage_bucket.pipeline_bucket.name
  source = "${path.module}/data/training_data.csv"
}

resource "google_storage_bucket_object" "pipeline_spec" {
  name           = "pipelines/pipeline.yaml"
  bucket         = google_storage_bucket.pipeline_bucket.name
  source         = "${path.module}/pipeline/pipeline.yaml"
  detect_md5hash = true
}

resource "google_storage_bucket_object" "predict_schema" {
  name   = "schemas/predict_schema.yaml"
  bucket = google_storage_bucket.pipeline_bucket.name
  source = "${path.module}/pipeline/predict_schema.yaml"
}


# --- Package Trigger Cloud Function ---
data "archive_file" "cf_source" {
  type        = "zip"
  source_dir  = "${path.module}/trigger_function"
  output_path = "${path.module}/trigger_function.zip"
}

resource "google_storage_bucket_object" "cf_source_zip" {
  name   = "trigger_function-${data.archive_file.cf_source.output_md5}.zip"
  bucket = google_storage_bucket.trigger_bucket.name
  source = data.archive_file.cf_source.output_path
}

# --- Service Accounts & Permissions ---

# Pipeline Execution Service Account
resource "google_service_account" "pipeline_sa" {
  account_id   = "anomaly-pipeline-sa"
  display_name = "Vertex AI Pipeline Execution Service Account"
  depends_on   = [google_project_service.services]
}

# Grant Project Owner role to Pipeline SA to fully unblock Vertex AI execution
resource "google_project_iam_member" "pipeline_sa_owner" {
  project = var.project_id
  role    = "roles/owner"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

# Vertex AI Service Agent Identity Creation (Forces GCP to instantiate it during deploy)
resource "google_project_service_identity" "ai_sa" {
  provider = google-beta
  project  = var.project_id
  service  = "aiplatform.googleapis.com"
  depends_on = [google_project_service.services]
}

# Bind BigQuery Admin to the Service Agent to allow querying logged serving traffic
resource "google_project_iam_member" "vertex_agent_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.admin"
  member  = "serviceAccount:${google_project_service_identity.ai_sa.email}"
}

# Bind Storage Object Viewer to the Service Agent to allow loading custom GCS baselines
resource "google_project_iam_member" "vertex_agent_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_project_service_identity.ai_sa.email}"
}

# Bind BigQuery Admin to the Model Monitoring Service Agent to allow querying prediction logs
resource "google_project_iam_member" "vertex_mm_agent_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.admin"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-vertex-mm.iam.gserviceaccount.com"
}

# Bind Storage Object Viewer to the Model Monitoring Service Agent to allow reading GCS baselines
resource "google_project_iam_member" "vertex_mm_agent_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-vertex-mm.iam.gserviceaccount.com"
}


