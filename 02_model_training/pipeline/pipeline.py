from kfp import dsl
from kfp import compiler

# Define the training component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def train_gpt2_custom_job(
    project_id: str,
    region: str,
    bucket_name: str,
    pipeline_sa_email: str,
    gpu_type: str,
    gpu_limit: int,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    weight_decay: float,
    model_output_uri: str,
    shuffle_buffer: int,
    dataset_subset: str,
    wandb_api_key: str,
    num_workers: int,
    dataset_bin: str,
    checkpoint_activations: str,
    tp_size: int,
    pp_size: int,
    scheduling_strategy: str = "ON_DEMAND",
):
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=region)
    
    # 1. Determine local GPUs per node and total node replicas
    if "A100_80GB" in gpu_type:
        if gpu_limit >= 8:
            machine_type = "a2-ultragpu-8g"
            max_gpus_per_node = 8
        elif gpu_limit >= 4:
            machine_type = "a2-ultragpu-4g"
            max_gpus_per_node = 4
        elif gpu_limit >= 2:
            machine_type = "a2-ultragpu-2g"
            max_gpus_per_node = 2
        else:
            machine_type = "a2-ultragpu-1g"
            max_gpus_per_node = 1
    elif "A100" in gpu_type:
        if gpu_limit >= 8:
            machine_type = "a2-highgpu-8g"
            max_gpus_per_node = 8
        elif gpu_limit >= 4:
            machine_type = "a2-highgpu-4g"
            max_gpus_per_node = 4
        elif gpu_limit >= 2:
            machine_type = "a2-highgpu-2g"
            max_gpus_per_node = 2
        else:
            machine_type = "a2-highgpu-1g"
            max_gpus_per_node = 1
    elif "L4" in gpu_type:
        if gpu_limit >= 8:
            machine_type = "g2-standard-96"
            max_gpus_per_node = 8
        elif gpu_limit >= 4:
            machine_type = "g2-standard-48"
            max_gpus_per_node = 4
        elif gpu_limit >= 2:
            machine_type = "g2-standard-24"
            max_gpus_per_node = 2
        else:
            machine_type = "g2-standard-12"
            max_gpus_per_node = 1
    else: # Default: NVIDIA_TESLA_T4
        max_gpus_per_node = 4
        machine_type = "n1-standard-16"
        
    gpus_per_node = min(gpu_limit, max_gpus_per_node)
    num_nodes = max(1, gpu_limit // gpus_per_node)
    
    print(f"Custom Job Config: {num_nodes} node(s) x {gpus_per_node} GPU(s) ({gpu_type}) using {machine_type} machine type.")

    # 2. Build the command/args for DDP run using the container's run_ddp.sh script
    command = ["/app/run_ddp.sh"]
    args = [
        f"--nproc_per_node={gpus_per_node}",
        "train.py",
        "--learning-rate", str(learning_rate),
        "--max-steps", str(max_steps),
        "--batch-size", str(batch_size),
        "--weight-decay", str(weight_decay),
        "--model-output-uri", model_output_uri,
        "--shuffle-buffer", str(shuffle_buffer),
        "--dataset-subset", dataset_subset,
        "--num-workers", str(num_workers),
        "--dataset-bin", dataset_bin,
        "--checkpoint-activations", checkpoint_activations,
        "--tp-size", str(tp_size),
        "--pp-size", str(pp_size),
    ]
    
    # 3. Create the Custom Job worker pool specifications
    worker_pool_specs = []
    
    master_container_spec = {
        "image_uri": f"us-central1-docker.pkg.dev/{project_id}/gpt2-prediction-images/gpt2-train:latest",
        "command": command,
        "args": args,
    }
    if wandb_api_key:
        master_container_spec["env"] = [{"name": "WANDB_API_KEY", "value": wandb_api_key}]

    # Pool 0: Master Node
    worker_pool_specs.append({
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": gpu_type,
            "accelerator_count": gpus_per_node,
        },
        "replica_count": 1,
        "container_spec": master_container_spec
    })
    
    # Pool 1: Worker Nodes
    if num_nodes > 1:
        worker_container_spec = {
            "image_uri": f"us-central1-docker.pkg.dev/{project_id}/gpt2-prediction-images/gpt2-train:latest",
            "command": command,
            "args": args,
        }
        if wandb_api_key:
            worker_container_spec["env"] = [{"name": "WANDB_API_KEY", "value": wandb_api_key}]

        worker_pool_specs.append({
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": gpu_type,
                "accelerator_count": gpus_per_node,
            },
            "replica_count": num_nodes - 1,
            "container_spec": worker_container_spec
        })
        
    print(f"Submitting CustomJob under service account: {pipeline_sa_email}")
    job = aiplatform.CustomJob(
        display_name="gpt2-custom-training-job-run",
        worker_pool_specs=worker_pool_specs,
        staging_bucket=f"gs://{bucket_name}/staging",
    )
    
    job.run(
        service_account=pipeline_sa_email,
        scheduling_strategy=scheduling_strategy
    )
    print("Training Job completed successfully.")


# Define the fine-tuning component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def finetune_gpt2_lora_job(
    project_id: str,
    region: str,
    bucket_name: str,
    pipeline_sa_email: str,
    base_model_uri: str,
    alpaca_json_uri: str,
    output_model_uri: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    wandb_api_key: str,
    lora_max_steps: int = -1,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    gpu_type = "NVIDIA_TESLA_T4"
    machine_type = "n1-standard-8"
    
    command = ["bash", "-c"]
    args_script = (
        f"python finetune_lora.py "
        f"--base-model-uri {base_model_uri} "
        f"--alpaca-json-uri {alpaca_json_uri} "
        f"--output-model-uri {output_model_uri} "
        f"--learning-rate {learning_rate} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--lora-rank {lora_rank} "
        f"--lora-alpha {lora_alpha} "
        f"--wandb-api-key '{wandb_api_key}'"
    )
    if lora_max_steps > 0:
        args_script += f" --max-steps {lora_max_steps}"
    args = [args_script]

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": gpu_type,
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": f"us-central1-docker.pkg.dev/{project_id}/gpt2-prediction-images/gpt2-train:latest",
            "command": command,
            "args": args,
        }
    }]

    print(f"Submitting LoRA Fine-tuning CustomJob under service account: {pipeline_sa_email}")
    job = aiplatform.CustomJob(
        display_name="gpt2-lora-finetuning-job-run",
        worker_pool_specs=worker_pool_specs,
        staging_bucket=f"gs://{bucket_name}/staging",
    )
    
    job.run(service_account=pipeline_sa_email)
    print("Fine-tuning Job completed successfully.")


# Define the deploy component
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_gpt2(
    project_id: str,
    region: str,
    model_display_name: str,
    endpoint_display_name: str,
    model_gcs_uri: str,
    serving_container_image_uri: str,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # 1. Handle Endpoint
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

    # 2. Upload Model
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )
    parent_model = models[0].resource_name if models else None

    print(f"Uploading model to registry: {model_display_name}")
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
    print(f"Uploaded model: {uploaded_model.resource_name}")

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

    # Undeploy old models
    for deployed_model in endpoint.list_models():
        if deployed_model.model != uploaded_model.resource_name:
            print(f"Undeploying old model deployment: {deployed_model.id}")
            endpoint.undeploy(deployed_model_id=deployed_model.id)
    print("Undeployment of older versions complete.")

# Define KFP Pipeline
@dsl.pipeline(
    name="gpt2-training-deployment-pipeline",
    description="Pipeline to train GPT-2 on small corpus, finetune on Alpaca JSON via LoRA, and deploy to Gemini Enterprise Agent Platform Endpoint"
)
def gpt2_pipeline(
    project_id: str,
    region: str,
    bucket_name: str,
    pipeline_sa_email: str,
    model_output_uri: str,
    model_display_name: str,
    endpoint_display_name: str,
    serving_container_image_uri: str,
    learning_rate: float = 5e-4,
    max_steps: int = 1000,
    batch_size: int = 8,
    weight_decay: float = 0.1,
    shuffle_buffer: int = 2000,
    dataset_subset: str = "sample-10BT",
    wandb_api_key: str = "",
    gpu_type: str = "NVIDIA_TESLA_A100",
    gpu_limit: int = 8,
    num_workers: int = 2,
    dataset_bin: str = "gs://<YOUR_GCS_BUCKET>/dataset/train.bin",
    cpu_limit: str = "8",
    memory_limit: str = "52G",
    checkpoint_activations: str = "False",
    tp_size: int = 1,
    pp_size: int = 1,
    alpaca_json_uri: str = "gs://<YOUR_GCS_BUCKET>/dataset/alpaca_data.json",
    lora_learning_rate: float = 2e-4,
    lora_epochs: int = 3,
    lora_batch_size: int = 4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    scheduling_strategy: str = "ON_DEMAND",
    lora_max_steps: int = -1,
):
    # Step 1: Pretraining (submits Gemini Enterprise Agent Platform CustomJob for multi-worker support)
    train_task = train_gpt2_custom_job(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        pipeline_sa_email=pipeline_sa_email,
        gpu_type=gpu_type,
        gpu_limit=gpu_limit,
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_size=batch_size,
        weight_decay=weight_decay,
        model_output_uri=model_output_uri,
        shuffle_buffer=shuffle_buffer,
        dataset_subset=dataset_subset,
        wandb_api_key=wandb_api_key,
        num_workers=num_workers,
        dataset_bin=dataset_bin,
        checkpoint_activations=checkpoint_activations,
        tp_size=tp_size,
        pp_size=pp_size,
        scheduling_strategy=scheduling_strategy,
    )
    train_task.set_caching_options(enable_caching=False)
    train_task.set_retry(num_retries=5, backoff_duration="60s", backoff_factor=2.0)

    # Step 2: LoRA Fine-tuning
    lora_model_dir = f"{model_output_uri}_lora"
    finetune_task = finetune_gpt2_lora_job(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        pipeline_sa_email=pipeline_sa_email,
        base_model_uri=f"{model_output_uri}/model.pth",
        alpaca_json_uri=alpaca_json_uri,
        output_model_uri=f"{lora_model_dir}/model.pth",
        learning_rate=lora_learning_rate,
        epochs=lora_epochs,
        batch_size=lora_batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        wandb_api_key=wandb_api_key,
        lora_max_steps=lora_max_steps,
    )
    finetune_task.set_caching_options(enable_caching=False)
    finetune_task.set_retry(num_retries=5, backoff_duration="60s", backoff_factor=2.0)
    finetune_task.after(train_task)

    # Step 3: Deploy
    deploy_task = deploy_gpt2(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
        model_gcs_uri=lora_model_dir,
        serving_container_image_uri=serving_container_image_uri
    )
    deploy_task.set_caching_options(enable_caching=False)
    deploy_task.after(finetune_task)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="pipeline.yaml", help="Path to output the compiled pipeline spec")
    args = parser.parse_args()
    
    compiler.Compiler().compile(
        pipeline_func=gpt2_pipeline,
        package_path=args.output
    )
    print(f"Compiled pipeline specification saved to {args.output}")
