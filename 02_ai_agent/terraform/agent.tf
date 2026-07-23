resource "google_dialogflow_cx_agent" "coordinator" {
  provider             = google-beta
  project              = var.project_id
  display_name         = "gcp-cloud-troubleshooter"
  location             = var.region
  default_language_code = "en"
  time_zone            = "America/New_York"
  
  description          = "Generative AI multi-agent platform for monitoring and troubleshooting GCP resources."
  
  depends_on = [google_project_service.services]
}
