import base64
import os
import json
from google.cloud import aiplatform

def trigger_retraining(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    print(f"Received Pub/Sub message: {pubsub_message}")
    
    # Try parsing as JSON to log details, but trigger regardless
    try:
        log_payload = json.loads(pubsub_message)
        print(f"Parsed log payload: {json.dumps(log_payload, indent=2)}")
    except Exception as e:
        print(f"Could not parse message as JSON: {e}")

    # Fetch configuration from environment variables
    project_id = os.environ.get("PROJECT_ID")
    region = os.environ.get("REGION")
    pipeline_yaml_gcs_uri = os.environ.get("PIPELINE_YAML_GCS_URI")
    training_data_gcs_uri = os.environ.get("TRAINING_DATA_GCS_URI")
    model_output_gcs_uri = os.environ.get("MODEL_OUTPUT_GCS_URI")
    model_display_name = os.environ.get("MODEL_DISPLAY_NAME")
    endpoint_display_name = os.environ.get("ENDPOINT_DISPLAY_NAME")
    serving_container_image_uri = os.environ.get("SERVING_CONTAINER_IMAGE_URI")
    deployed_model_id = os.environ.get("DEPLOYED_MODEL_ID")
    skew_threshold_str = os.environ.get("SKEW_THRESHOLD")
    staging_bucket = os.environ.get("STAGING_BUCKET")
    pipeline_service_account = os.environ.get("PIPELINE_SERVICE_ACCOUNT")
    predict_schema_gcs_uri = os.environ.get("PREDICT_SCHEMA_GCS_URI")
    bigquery_table_uri = os.environ.get("BIGQUERY_TABLE_URI", "")

    if not all([project_id, region, pipeline_yaml_gcs_uri, training_data_gcs_uri, model_output_gcs_uri, model_display_name, endpoint_display_name, serving_container_image_uri, deployed_model_id, skew_threshold_str, staging_bucket, pipeline_service_account, predict_schema_gcs_uri]):
        missing = [k for k, v in {
            "PROJECT_ID": project_id,
            "REGION": region,
            "PIPELINE_YAML_GCS_URI": pipeline_yaml_gcs_uri,
            "TRAINING_DATA_GCS_URI": training_data_gcs_uri,
            "MODEL_OUTPUT_GCS_URI": model_output_gcs_uri,
            "MODEL_DISPLAY_NAME": model_display_name,
            "ENDPOINT_DISPLAY_NAME": endpoint_display_name,
            "SERVING_CONTAINER_IMAGE_URI": serving_container_image_uri,
            "DEPLOYED_MODEL_ID": deployed_model_id,
            "SKEW_THRESHOLD": skew_threshold_str,
            "STAGING_BUCKET": staging_bucket,
            "PIPELINE_SERVICE_ACCOUNT": pipeline_service_account,
            "PREDICT_SCHEMA_GCS_URI": predict_schema_gcs_uri
        }.items() if not v]
        raise ValueError(f"Missing required environment variables: {missing}")

    skew_threshold = float(skew_threshold_str)

    print(f"Initializing Vertex AI SDK for project {project_id} in region {region} with staging bucket {staging_bucket}")
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    print(f"Submitting Pipeline Job from template: {pipeline_yaml_gcs_uri}")
    job = aiplatform.PipelineJob(
        display_name="retrained-anomaly-pipeline",
        template_path=pipeline_yaml_gcs_uri,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "training_data_gcs_uri": training_data_gcs_uri,
            "model_output_gcs_uri": model_output_gcs_uri,
            "model_display_name": model_display_name,
            "endpoint_display_name": endpoint_display_name,
            "serving_container_image_uri": serving_container_image_uri,
            "deployed_model_id": deployed_model_id,
            "skew_threshold": skew_threshold,
            "predict_schema_gcs_uri": predict_schema_gcs_uri,
            "bigquery_table_uri": bigquery_table_uri,
        },
        enable_caching=False,
    )

    job.run(service_account=pipeline_service_account, sync=False)
    print("Pipeline Job submitted successfully.")
