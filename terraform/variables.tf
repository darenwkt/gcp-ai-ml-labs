variable "project_id" {
  type        = string
  description = "The GCP project ID to deploy resources into."
  default     = "darenwkt-sandbox"
}


variable "region" {
  type        = string
  description = "The GCP region to deploy resources in."
  default     = "us-central1"
}

variable "endpoint_display_name" {
  type        = string
  description = "The display name of the Vertex AI Endpoint."
  default     = "anomaly-detection-endpoint"
}

variable "model_display_name" {
  type        = string
  description = "The display name of the Vertex AI Model."
  default     = "anomaly-detection-model"
}

variable "deployed_model_id" {
  type        = string
  description = "The user-specified ID of the deployed model on the Endpoint."
  default     = "anomaly-detector-deployed"
}

variable "serving_container_image_uri" {
  type        = string
  description = "The container image URI used for predictions."
  default     = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
}

variable "skew_threshold" {
  type        = number
  description = "The threshold for training-serving skew detection (L1 statistical distance)."
  default     = 0.01
}

