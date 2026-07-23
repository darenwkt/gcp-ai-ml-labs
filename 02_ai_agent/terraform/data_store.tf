resource "google_discovery_engine_data_store" "gcp_docs" {
  provider            = google-beta
  project             = var.project_id
  location            = var.data_store_location
  data_store_id       = "gcp-documentation-store"
  display_name        = "GCP Documentation Store"
  industry_vertical   = "GENERIC"
  content_config      = "CONTENT_REQUIRED"
  solution_types      = ["SOLUTION_TYPE_SEARCH"]
  
  depends_on = [google_project_service.services]
}
