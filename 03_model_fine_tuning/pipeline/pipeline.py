import os
from kfp import dsl

# 1. Custom Training Job Component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def train_gpt2_sft_job(
    project_id: str,
    region: str,
    bucket_name: str,
    pipeline_sa_email: str,
    model_output_uri: str,
    training_image_uri: str,
    model_id: str,
    dataset_name: str,
    dataset_config: str,
    dataset_csv_gcs: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    max_steps: int = -1,
    finetuning_type: str = "lora",
) -> str:
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=region)
    
    command = ["python", "finetune.py"]
    args = [
        "--project-id", project_id,
        "--model-id", model_id,
        "--dataset-name", dataset_name,
        "--dataset-config", dataset_config,
        "--dataset-csv-gcs", dataset_csv_gcs,
        "--output-model-gcs", model_output_uri,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--max-steps", str(max_steps),
        "--finetuning-type", finetuning_type,
    ]
    
    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": "g2-standard-8",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": training_image_uri,
            "command": command,
            "args": args,
        }
    }]
    
    print(f"Submitting Fine-Tuning CustomJob under service account: {pipeline_sa_email}")
    job = aiplatform.CustomJob(
        display_name="gpt2-sft-finetuning-custom-job",
        worker_pool_specs=worker_pool_specs,
        staging_bucket=f"gs://{bucket_name}/staging",
    )
    job.run(service_account=pipeline_sa_email)
    print(f"CustomJob finished successfully! Artifacts are in: {model_output_uri}")
    return model_output_uri

# 2. Deploy Model to Endpoint Component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model_to_endpoint(
    project_id: str,
    region: str,
    model_display_name: str,
    endpoint_display_name: str,
    model_gcs_uri: str,
    serving_container_image_uri: str,
) -> str:
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=region)
    
    # Check if endpoint exists, otherwise create it
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
            project=project_id,
            location=region
        )
        print(f"Created endpoint: {endpoint.resource_name}")
        
    # Check if model exists to register new version
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )
    parent_model = models[0].resource_name if models else None
    
    print(f"Uploading model to registry from {model_gcs_uri}...")
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_gcs_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/healthz",
        parent_model=parent_model,
        is_default_version=True,
    )
    print(f"Uploaded model: {uploaded_model.resource_name}")
    
    # Deploy model to endpoint (100% traffic, CPU-based serving)
    print(f"Deploying model to endpoint {endpoint.resource_name}...")
    endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name=model_display_name,
        traffic_percentage=100,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1
    )
    print("Model deployed successfully.")
    return endpoint.resource_name

# 3. Pipeline Definition
@dsl.pipeline(
    name="gpt2-sft-support-tickets-pipeline",
    description="Fine-tune and deploy a model to respond to IT support tickets."
)
def gpt2_sft_pipeline(
    project_id: str,
    region: str,
    bucket_name: str,
    pipeline_sa_email: str,
    model_output_uri: str,
    training_image_uri: str,
    serving_container_image_uri: str,
    model_display_name: str = "gpt2-it-support-model",
    endpoint_display_name: str = "gpt2-it-support-endpoint",
    model_id: str = "gpt2",
    dataset_name: str = "enzo-joseph/customer-support-tickets",
    dataset_config: str = "en",
    dataset_csv_gcs: str = "",
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_steps: int = -1,
    finetuning_type: str = "lora",
):
    train_task = train_gpt2_sft_job(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        pipeline_sa_email=pipeline_sa_email,
        model_output_uri=model_output_uri,
        training_image_uri=training_image_uri,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_csv_gcs=dataset_csv_gcs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_steps=max_steps,
        finetuning_type=finetuning_type,
    )
    
    deploy_task = deploy_model_to_endpoint(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
        model_gcs_uri=train_task.output,
        serving_container_image_uri=serving_container_image_uri,
    )
