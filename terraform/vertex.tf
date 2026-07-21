resource "google_vertex_ai_endpoint" "endpoint" {
  name         = "184013890"
  display_name = var.endpoint_display_name
  location     = var.region
  project      = var.project_id
  
  labels = local.common_labels

  predict_request_response_logging_config {
    enabled       = true
    sampling_rate = 1.0
    bigquery_destination {
      output_uri = "bq://${var.project_id}.prediction_logs.endpoint_184013890_logs"
    }
  }

  depends_on = [google_project_service.services]
}

# Grant the Vertex AI Service Agent ActAs permission on the Pipeline SA
resource "google_service_account_iam_member" "vertex_agent_act_as_pipeline_sa" {
  service_account_id = google_service_account.pipeline_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}
