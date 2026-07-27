import sys
from google.cloud import aiplatform

def redeploy():
    import os
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "<YOUR_GCP_PROJECT_ID>")
    region = "us-central1"
    model_display_name = "gpt2-text-generation-model-ddp-8xa100"
    endpoint_display_name = "gpt2-serving-endpoint-ddp-8xa100"
    model_gcs_uri = os.environ.get(
        "MODEL_GCS_URI", 
        f"gs://{project_id}-gpt2-pipeline-artifacts/model-output/20260718-103945"
    )
    serving_container_image_uri = os.environ.get(
        "SERVING_IMAGE_URI",
        f"us-central1-docker.pkg.dev/{project_id}/gpt2-prediction-images/gpt2-predict:latest"
    )

    print("Initializing Gemini Enterprise Agent Platform SDK...")
    aiplatform.init(project=project_id, location=region)

    # 1. Get endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc"
    )
    if not endpoints:
        print(f"Creating new endpoint: {endpoint_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=region
        )
        print(f"Created endpoint: {endpoint.resource_name}")
    else:
        endpoint = endpoints[0]
        print(f"Found endpoint: {endpoint.resource_name}")

    # 2. Upload Model
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )
    parent_model = models[0].resource_name if models else None

    print(f"Uploading new model version to registry: {model_display_name}")
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_gcs_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/healthz",
        parent_model=parent_model,
        is_default_version=True,
    )
    print(f"Uploaded model version: {uploaded_model.resource_name} (Version ID: {uploaded_model.version_id})")

    # 3. Deploy Model to Endpoint
    print(f"Deploying model {uploaded_model.resource_name} to endpoint {endpoint.resource_name}...")
    endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name=model_display_name,
        traffic_percentage=100,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
    )
    print("Model deployed to endpoint successfully.")

    # 4. Undeploy old models
    for deployed_model in endpoint.list_models():
        if deployed_model.model != uploaded_model.resource_name:
            print(f"Undeploying old model deployment: {deployed_model.id}")
            endpoint.undeploy(deployed_model_id=deployed_model.id)
    print("Undeployment of older versions complete.")

if __name__ == "__main__":
    redeploy()
