variable "project_id" {
  type        = string
  description = "The GCP project ID to deploy resources into."
}

variable "region" {
  type        = string
  description = "The GCP region to deploy Dialogflow CX and Discovery Engine (e.g. global, us-central1)."
  default     = "global"
}

variable "data_store_location" {
  type        = string
  description = "The location of the Discovery Engine data store (e.g. global, us, eu)."
  default     = "global"
}
