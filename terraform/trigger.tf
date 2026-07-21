# --- Pub/Sub Topic ---
resource "google_pubsub_topic" "retraining_topic" {
  name       = "anomaly-retraining-alerts"
  labels     = local.common_labels
  depends_on = [google_project_service.services]
}

# --- Cloud Logging Router (Sink) ---
# Routes model deployment monitoring alert logs to the Pub/Sub topic
resource "google_logging_project_sink" "monitoring_sink" {
  name        = "anomaly-monitoring-log-sink"
  destination = "pubsub.googleapis.com/${google_pubsub_topic.retraining_topic.id}"
  filter      = "(resource.type=\"aiplatform.googleapis.com/ModelMonitor\" AND jsonPayload.@type=\"type.googleapis.com/google.cloud.aiplatform.logging.ModelMonitoringJobAnomaliesLogEntry\") OR (resource.type=\"model_monitoring_job\" AND jsonPayload.modelMonitoringAnomaly=true)"

  unique_writer_identity = true
}

# Grant the logging sink service account publisher permissions on the Pub/Sub topic
resource "google_pubsub_topic_iam_member" "sink_publisher" {
  topic  = google_pubsub_topic.retraining_topic.name
  role   = "roles/pubsub.publisher"
  member = google_logging_project_sink.monitoring_sink.writer_identity
}

# --- Cloud Function Service Account ---
resource "google_service_account" "function_sa" {
  account_id   = "anomaly-trigger-sa"
  display_name = "Retraining Trigger Cloud Function Service Account"
  depends_on   = [google_project_service.services]
}

# Grant Project Owner role to Cloud Function SA to fully unblock retraining triggers
resource "google_project_iam_member" "function_sa_owner" {
  project = var.project_id
  role    = "roles/owner"
  member  = "serviceAccount:${google_service_account.function_sa.email}"
}

# --- Cloud Function (Gen 2) ---
resource "google_cloudfunctions2_function" "retrain_function" {
  name        = "retrain-trigger-function"
  location    = var.region
  description = "Triggered by Pub/Sub model monitoring alerts to start Vertex AI retraining pipeline."
  labels      = local.common_labels

  build_config {
    runtime     = "python310"
    entry_point = "trigger_retraining"
    
    source {
      storage_source {
        bucket = google_storage_bucket.trigger_bucket.name
        object = google_storage_bucket_object.cf_source_zip.name
      }
    }
  }

  service_config {
    max_instance_count = 3
    min_instance_count = 0
    available_memory   = "256Mi"
    timeout_seconds    = 60
    
    service_account_email = google_service_account.function_sa.email

    environment_variables = {
      PROJECT_ID                  = var.project_id
      REGION                      = var.region
      PIPELINE_YAML_GCS_URI       = "gs://${google_storage_bucket.pipeline_bucket.name}/${google_storage_bucket_object.pipeline_spec.name}"
      TRAINING_DATA_GCS_URI       = "gs://${google_storage_bucket.pipeline_bucket.name}/${google_storage_bucket_object.training_data.name}"
      MODEL_OUTPUT_GCS_URI        = "gs://${google_storage_bucket.pipeline_bucket.name}/model-output"
      MODEL_DISPLAY_NAME          = var.model_display_name
      ENDPOINT_DISPLAY_NAME       = var.endpoint_display_name
      SERVING_CONTAINER_IMAGE_URI = var.serving_container_image_uri
      DEPLOYED_MODEL_ID           = var.deployed_model_id
      SKEW_THRESHOLD              = tostring(var.skew_threshold)
      STAGING_BUCKET              = "gs://${google_storage_bucket.pipeline_bucket.name}/staging"
      PIPELINE_SERVICE_ACCOUNT    = google_service_account.pipeline_sa.email
      PREDICT_SCHEMA_GCS_URI      = "gs://${google_storage_bucket.pipeline_bucket.name}/${google_storage_bucket_object.predict_schema.name}"
      BIGQUERY_TABLE_URI          = "${var.project_id}.prediction_logs.endpoint_184013890_logs"
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.retraining_topic.id
    retry_policy   = "RETRY_POLICY_DO_NOT_RETRY"
  }

  depends_on = [
    google_project_iam_member.function_sa_owner,
    google_pubsub_topic_iam_member.sink_publisher
  ]
}

data "google_project" "project" {}

resource "google_cloud_run_service_iam_member" "cf_invoker" {
  location = google_cloudfunctions2_function.retrain_function.location
  project  = google_cloudfunctions2_function.retrain_function.project
  service  = google_cloudfunctions2_function.retrain_function.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}


