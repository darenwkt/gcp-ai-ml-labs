variable "project_id" {
  type        = string
  description = "The GCP project ID to deploy resources into."
}

variable "region" {
  type        = string
  description = "The GCP region to deploy resources in."
  default     = "us-central1"
}

variable "endpoint_display_name" {
  type        = string
  description = "The display name of the Gemini Enterprise Agent Platform Endpoint."
  default     = "gpt2-serving-endpoint-ddp-8xa100"
}

variable "model_display_name" {
  type        = string
  description = "The display name of the Gemini Enterprise Agent Platform Model."
  default     = "gpt2-text-generation-model-ddp-8xa100"
}

variable "repository_name" {
  type        = string
  description = "The name of the Artifact Registry repository for custom containers."
  default     = "gpt2-prediction-images"
}
