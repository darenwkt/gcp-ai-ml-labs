variable "project_id" {
  type        = string
  description = "The GCP Project ID where resources will be provisioned."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "The GCP region for Vertex AI and other services."
}

variable "bucket_name" {
  type        = string
  description = "The name of the GCS bucket to store pipeline artifacts."
}

variable "pipeline_sa_name" {
  type        = string
  default     = "gpt2-finetune-pipeline-sa"
  description = "The custom service account name to execute the fine-tuning pipeline."
}

variable "repository_id" {
  type        = string
  default     = "gpt2-finetuning-images"
  description = "Artifact Registry repository ID for training and serving docker images."
}
