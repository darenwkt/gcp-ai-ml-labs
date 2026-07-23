terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.0.0"
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
    "dialogflow.googleapis.com",
    "discoveryengine.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbilling.googleapis.com",
    "securitycenter.googleapis.com"
  ]

  common_labels = {
    managed_by  = "terraform"
    project     = "multi-agent-ops-troubleshooter"
    environment = "sandbox"
  }
}

resource "google_project_service" "services" {
  for_each                   = toset(local.apis)
  project                    = var.project_id
  service                    = each.key
  disable_dependent_services = false
  disable_on_destroy         = false
}
