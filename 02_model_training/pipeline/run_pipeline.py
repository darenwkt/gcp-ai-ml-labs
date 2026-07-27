import os
import sys
import argparse
import tempfile
from google.cloud import aiplatform
from kfp import compiler

# Make parent directory importable and remove script's directory to avoid package collision
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]
sys.path.insert(0, parent_dir)

from pipeline.pipeline import gpt2_pipeline

def get_env_var(name):
    # Lightweight line parser for local .env files
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith(f"{name}="):
                    return clean_line.split("=", 1)[1]
    return os.environ.get(name, "")

def check_machine_type_availability(project_id: str, region: str, machine_type: str, gpu_type: str = None) -> bool:
    import google.auth
    import google.auth.transport.requests
    import requests

    credentials, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    headers = {"Authorization": f"Bearer {credentials.token}"}

    # 1. Get zones in region
    region_url = f"https://compute.googleapis.com/compute/v1/projects/{project_id}/regions/{region}"
    resp = requests.get(region_url, headers=headers)
    if resp.status_code != 200:
        print(f"Precheck warning: Failed to list zones in region {region} ({resp.status_code}): {resp.text}")
        return True

    zones = [z.split("/")[-1] for z in resp.json().get("zones", [])]

    # 2. Check machine type and its accelerator compatibility in each zone
    machine_available = False
    requested_gpu_supported = False
    
    gcp_gpu_name = gpu_type.lower().replace("_", "-") if gpu_type else None

    for zone in zones:
        mt_url = f"https://compute.googleapis.com/compute/v1/projects/{project_id}/zones/{zone}/machineTypes/{machine_type}"
        resp = requests.get(mt_url, headers=headers)
        if resp.status_code != 200:
            continue
        
        machine_available = True
        mt_data = resp.json()
        
        static_accels = mt_data.get("accelerators")
        if static_accels:
            if not gcp_gpu_name:
                requested_gpu_supported = True
            else:
                for accel in static_accels:
                    if accel.get("guestAcceleratorType") == gcp_gpu_name:
                        requested_gpu_supported = True
                        break
            if requested_gpu_supported:
                break
        else:
            if not gcp_gpu_name:
                requested_gpu_supported = True
                break
            else:
                accel_url = f"https://compute.googleapis.com/compute/v1/projects/{project_id}/zones/{zone}/acceleratorTypes/{gcp_gpu_name}"
                if requests.get(accel_url, headers=headers).status_code == 200:
                    requested_gpu_supported = True
                    break

    if not machine_available:
        print(f"Precheck error: Machine type '{machine_type}' is not available in region '{region}'.")
        return False

    if gcp_gpu_name and not requested_gpu_supported:
        print(f"Precheck error: Accelerator '{gpu_type}' is not compatible or available with machine type '{machine_type}' in region '{region}'.")
        return False

    return True

def submit_pipeline(
    project_id: str,
    region: str,
    bucket_name: str,
    model_display_name: str,
    endpoint_display_name: str,
    serving_container_image_uri: str,
    pipeline_sa_email: str,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    shuffle_buffer: int,
    dataset_subset: str,
    wandb_api_key: str,
    gpu_type: str,
    gpu_limit: int,
    num_workers: int,
    dataset_bin: str,
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
    # Resolve placeholders using actual bucket_name
    if "<YOUR_GCS_BUCKET>" in dataset_bin:
        dataset_bin = dataset_bin.replace("<YOUR_GCS_BUCKET>", bucket_name)
    if "<YOUR_GCS_BUCKET>" in alpaca_json_uri:
        alpaca_json_uri = alpaca_json_uri.replace("<YOUR_GCS_BUCKET>", bucket_name)

    # Auto-upload local alpaca_data.json if needed
    if alpaca_json_uri.startswith("gs://"):
        local_alpaca = "alpaca_data.json"
        if os.path.exists(local_alpaca):
            from google.cloud import storage
            from urllib.parse import urlparse
            parsed = urlparse(alpaca_json_uri)
            bucket_name_gcs = parsed.netloc
            blob_name = parsed.path.lstrip("/")
            
            print("Checking if Alpaca dataset exists in GCS...")
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name_gcs)
            blob = bucket.blob(blob_name)
            if not blob.exists():
                print(f"Uploading local {local_alpaca} to {alpaca_json_uri}...")
                blob.upload_from_filename(local_alpaca)
                print("Upload complete.")

    print("Initializing Gemini Enterprise Agent Platform client...")
    aiplatform.init(project=project_id, location=region)

    # Determine the pretraining machine type dynamically
    if "A100_80GB" in gpu_type:
        if gpu_limit >= 8:
            pretrain_machine_type = "a2-ultragpu-8g"
        elif gpu_limit >= 4:
            pretrain_machine_type = "a2-ultragpu-4g"
        elif gpu_limit >= 2:
            pretrain_machine_type = "a2-ultragpu-2g"
        else:
            pretrain_machine_type = "a2-ultragpu-1g"
    elif "A100" in gpu_type:
        if gpu_limit >= 8:
            pretrain_machine_type = "a2-highgpu-8g"
        elif gpu_limit >= 4:
            pretrain_machine_type = "a2-highgpu-4g"
        elif gpu_limit >= 2:
            pretrain_machine_type = "a2-highgpu-2g"
        else:
            pretrain_machine_type = "a2-highgpu-1g"
    elif "L4" in gpu_type:
        if gpu_limit >= 8:
            pretrain_machine_type = "g2-standard-96"
        elif gpu_limit >= 4:
            pretrain_machine_type = "g2-standard-48"
        elif gpu_limit >= 2:
            pretrain_machine_type = "g2-standard-24"
        else:
            pretrain_machine_type = "g2-standard-12"
    else: # Default: NVIDIA_TESLA_T4
        pretrain_machine_type = "n1-standard-16"

    print(f"Running resource precheck for pretraining machine type '{pretrain_machine_type}' and GPU '{gpu_type}' in region '{region}'...")
    if not check_machine_type_availability(project_id, region, pretrain_machine_type, gpu_type):
        raise ValueError(f"Pretraining resource precheck failed. Selected combination of machine type '{pretrain_machine_type}' and GPU '{gpu_type}' is not supported/available in region '{region}'.")

    print(f"Running resource precheck for fine-tuning machine type 'n1-standard-8' and GPU 'NVIDIA_TESLA_T4' in region '{region}'...")
    if not check_machine_type_availability(project_id, region, "n1-standard-8", "NVIDIA_TESLA_T4"):
        raise ValueError(f"Fine-tuning resource precheck failed. Selected combination of machine type 'n1-standard-8' and GPU 'NVIDIA_TESLA_T4' is not supported/available in region '{region}'.")

    print("Resource precheck passed successfully.")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_output_uri = f"gs://{bucket_name}/model-output/{timestamp}"
    pipeline_root = f"gs://{bucket_name}/pipeline_root"

    # Compute resource limits dynamically based on GPU type and limit
    if "A100" in gpu_type:
        if gpu_limit >= 8:
            cpu_limit = "96"
            memory_limit = "680G"
        else:
            cpu_limit = "12"
            memory_limit = "85G"
    elif "L4" in gpu_type:
        cpu_limit = "16"
        memory_limit = "96G"
    else:
        cpu_limit = "8"
        memory_limit = "52G"

    print(f"Compiling pipeline dynamically for {gpu_limit}x {gpu_type} (CPU limit: {cpu_limit}, Memory limit: {memory_limit})...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_yaml = os.path.join(tmpdir, "pipeline.yaml")
        compiler.Compiler().compile(
            pipeline_func=gpt2_pipeline,
            package_path=temp_yaml
        )
        
        # Read compiled YAML and disable caching manually
        with open(temp_yaml, "r") as f:
            yaml_content = f.read()
            
        # Replace cachingOptions: {} with enableCache: true
        # Make sure to indent correctly by 10 spaces
        yaml_content = yaml_content.replace(
            "cachingOptions: {}",
            "cachingOptions:\n          enableCache: true"
        )
        
        with open(temp_yaml, "w") as f:
            f.write(yaml_content)

        print("Creating PipelineJob...")
        job = aiplatform.PipelineJob(
            display_name="gpt2-pipeline-job",
            template_path=temp_yaml,
            pipeline_root=pipeline_root,
            parameter_values={
                "project_id": project_id,
                "region": region,
                "bucket_name": bucket_name,
                "pipeline_sa_email": pipeline_sa_email,
                "model_output_uri": model_output_uri,
                "model_display_name": model_display_name,
                "endpoint_display_name": endpoint_display_name,
                "serving_container_image_uri": serving_container_image_uri,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "weight_decay": 0.1,
                "shuffle_buffer": shuffle_buffer,
                "dataset_subset": dataset_subset,
                "wandb_api_key": wandb_api_key,
                "gpu_type": gpu_type,
                "gpu_limit": gpu_limit,
                "num_workers": num_workers,
                "dataset_bin": dataset_bin,
                "cpu_limit": cpu_limit,
                "memory_limit": memory_limit,
                "checkpoint_activations": checkpoint_activations,
                "tp_size": tp_size,
                "pp_size": pp_size,
                "alpaca_json_uri": alpaca_json_uri,
                "lora_learning_rate": lora_learning_rate,
                "lora_epochs": lora_epochs,
                "lora_batch_size": lora_batch_size,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "scheduling_strategy": scheduling_strategy,
                "lora_max_steps": lora_max_steps,
            }
        )

        print(f"Submitting pipeline to run under service account: {pipeline_sa_email}...")
        job.submit(
            service_account=pipeline_sa_email
        )
        print("Pipeline submitted successfully!")
        print(f"Pipeline job link: {job.resource_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--model-display-name", type=str, default="gpt2-text-generation-model-ddp-8xa100")
    parser.add_argument("--endpoint-display-name", type=str, default="gpt2-serving-endpoint-ddp-8xa100")
    parser.add_argument("--serving-image", type=str, required=True)
    parser.add_argument("--pipeline-sa", type=str, required=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=2000)
    parser.add_argument("--dataset-subset", type=str, default="sample-10BT")
    
    # Auto-read API key from local .env
    default_wandb_key = get_env_var("WANDB_API_KEY")
    parser.add_argument("--wandb-key", type=str, default=default_wandb_key, help="Weights & Biases API Key")
    parser.add_argument("--gpu-type", type=str, default="NVIDIA_TESLA_A100", help="GCP GPU Accelerator Type (e.g. NVIDIA_TESLA_T4, NVIDIA_L4, NVIDIA_TESLA_A100)")
    parser.add_argument("--gpu-limit", type=int, default=8, help="Number of GPU accelerators to allocate")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader worker processes")
    parser.add_argument("--dataset-bin", type=str, default="gs://<YOUR_GCS_BUCKET>/dataset/train.bin", help="GCS URI or local path to pre-tokenized train.bin")
    parser.add_argument("--checkpoint-activations", type=str, default="False", help="Enable activation checkpointing (True/False)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline Parallelism size")
    parser.add_argument("--scheduling-strategy", type=str, default="ON_DEMAND", help="Scheduling strategy (ON_DEMAND, FLEX_START, SPOT)")
    
    # LoRA / SFT arguments
    parser.add_argument("--alpaca-json-uri", type=str, default="", help="GCS URI to alpaca_data.json. Defaults to gs://<bucket-name>/dataset/alpaca_data.json")
    parser.add_argument("--lora-lr", type=float, default=2e-4, help="Learning rate for LoRA fine-tuning")
    parser.add_argument("--lora-epochs", type=int, default=3, help="Epochs for LoRA fine-tuning")
    parser.add_argument("--lora-batch-size", type=int, default=4, help="Batch size for LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-max-steps", type=int, default=-1, help="Max training steps for LoRA fine-tuning")

    args = parser.parse_args()

    alpaca_uri = args.alpaca_json_uri if args.alpaca_json_uri else f"gs://{args.bucket_name}/dataset/alpaca_data.json"

    submit_pipeline(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket_name,
        model_display_name=args.model_display_name,
        endpoint_display_name=args.endpoint_display_name,
        serving_container_image_uri=args.serving_image,
        pipeline_sa_email=args.pipeline_sa,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        dataset_subset=args.dataset_subset,
        wandb_api_key=args.wandb_key,
        gpu_type=args.gpu_type,
        gpu_limit=args.gpu_limit,
        num_workers=args.num_workers,
        dataset_bin=args.dataset_bin,
        checkpoint_activations=args.checkpoint_activations,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        alpaca_json_uri=alpaca_uri,
        lora_learning_rate=args.lora_lr,
        lora_epochs=args.lora_epochs,
        lora_batch_size=args.lora_batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        scheduling_strategy=args.scheduling_strategy,
        lora_max_steps=args.lora_max_steps,
    )
