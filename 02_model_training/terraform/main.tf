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

locals {
  apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com"
  ]

  common_labels = {
    managed_by  = "terraform"
    project     = "gpt2-training-deployment"
    environment = "sandbox"
  }
}

# --- Project Services / APIs ---
resource "google_project_service" "services" {
  for_each                   = toset(local.apis)
  project                    = var.project_id
  service                    = each.key
  disable_dependent_services = false
  disable_on_destroy         = false
}

# --- Storage Bucket for Pipeline Artifacts ---
resource "google_storage_bucket" "pipeline_bucket" {
  name                        = "${var.project_id}-gpt2-pipeline-artifacts"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
  
  labels = local.common_labels

  depends_on = [google_project_service.services]
}

# --- Artifact Registry for Prediction Container ---
resource "google_artifact_registry_repository" "registry" {
  location      = var.region
  repository_id = var.repository_name
  description   = "Docker repository for GPT-2 prediction serving container"
  format        = "DOCKER"
  
  labels = local.common_labels

  depends_on = [google_project_service.services]
}

# --- Service Accounts & Permissions ---

# Pipeline Execution Service Account
resource "google_service_account" "pipeline_sa" {
  account_id   = "gpt2-pipeline-sa"
  display_name = "GPT-2 Pipeline Execution Service Account"
  depends_on   = [google_project_service.services]
}

# Grant Project Owner role to Pipeline SA to fully unblock Gemini Enterprise Agent Platform execution
resource "google_project_iam_member" "pipeline_sa_owner" {
  project = var.project_id
  role    = "roles/owner"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

# Get project data
data "google_project" "project" {}

# Gemini Enterprise Agent Platform Service Agent Identity Creation (Forces GCP to instantiate it during deploy)
resource "google_project_service_identity" "ai_sa" {
  provider = google-beta
  project  = var.project_id
  service  = "aiplatform.googleapis.com"
  depends_on = [google_project_service.services]
}

# Grant the Gemini Enterprise Agent Platform Service Agent ActAs permission on the Pipeline SA
resource "google_service_account_iam_member" "vertex_agent_act_as_pipeline_sa" {
  service_account_id = google_service_account.pipeline_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}
