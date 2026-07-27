import argparse
from google.cloud import aiplatform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--serving-image", type=str, required=True, help="URI to prediction Flask serving Docker container image")
    args = parser.parse_args()
    
    print("Initializing Vertex AI Client...")
    aiplatform.init(project=args.project_id, location=args.region)
    
    # 1. Create/Retrieve Endpoint
    endpoint_display_name = "gpt2-pretrained-endpoint"
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc"
    )
    if endpoints:
        endpoint = endpoints[0]
        print(f"Found existing endpoint: {endpoint.resource_name}")
    else:
        print(f"Creating new endpoint: {endpoint_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=args.project_id,
            location=args.region
        )
        print(f"Created endpoint: {endpoint.resource_name}")
        
    # 2. Upload Model
    model_display_name = "gpt2-pretrained-model"
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )
    parent_model = models[0].resource_name if models else None
    
    print("Uploading pretrained model to registry...")
    # NOTE: Since artifact_uri is omitted, the serving container will fall back to downloading
    # the baseline, raw gpt2 model from Hugging Face instead of reading fine-tuned adapters.
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        serving_container_image_uri=args.serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/healthz",
        parent_model=parent_model,
        is_default_version=True,
    )
    print(f"Uploaded model: {uploaded_model.resource_name}")
    
    # 3. Deploy Model to Endpoint
    print(f"Deploying model to endpoint {endpoint.resource_name}...")
    endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name=model_display_name,
        traffic_percentage=100,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1
    )
    print("Pretrained model deployed successfully!")
    print(f"Endpoint ID: {endpoint.name}")

if __name__ == "__main__":
    main()
