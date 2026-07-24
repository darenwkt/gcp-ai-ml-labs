import os
import sys
import argparse
import tempfile
import datetime
from google.cloud import aiplatform
from kfp import compiler

# Make parent directory importable and remove script's directory to avoid package collision
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]
sys.path.insert(0, parent_dir)

from pipeline.pipeline import gpt2_sft_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--pipeline-sa", type=str, required=True, help="Custom service account email for running pipeline steps")
    parser.add_argument("--model-id", type=str, default="gpt2")
    parser.add_argument("--dataset-name", type=str, default="enzo-joseph/customer-support-tickets")
    parser.add_argument("--dataset-config", type=str, default="en")
    parser.add_argument("--dataset-csv-gcs", type=str, default="", help="Optional GCS URI to custom dataset CSV")
    
    parser.add_argument("--training-image", type=str, required=True, help="URI to fine-tuning Docker container image")
    parser.add_argument("--serving-image", type=str, required=True, help="URI to prediction Flask serving Docker container image")
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    
    args = parser.parse_args()
    
    print("Initializing Vertex AI Client...")
    aiplatform.init(project=args.project_id, location=args.region)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_output_uri = f"gs://{args.bucket_name}/model-output/{timestamp}"
    pipeline_root = f"gs://{args.bucket_name}/pipeline_root"
    
    print("Compiling pipeline definition...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_yaml = os.path.join(tmpdir, "pipeline.yaml")
        compiler.Compiler().compile(
            pipeline_func=gpt2_sft_pipeline,
            package_path=temp_yaml
        )
        
        # Read compiled YAML and disable caching manually
        with open(temp_yaml, "r") as f:
            yaml_content = f.read()
            
        yaml_content = yaml_content.replace(
            "cachingOptions: {}",
            "cachingOptions:\n          enableCache: true"
        )
        
        with open(temp_yaml, "w") as f:
            f.write(yaml_content)
            
        print("Creating PipelineJob...")
        job = aiplatform.PipelineJob(
            display_name=f"gpt2-sft-support-tickets-run-{timestamp}",
            template_path=temp_yaml,
            pipeline_root=pipeline_root,
            parameter_values={
                "project_id": args.project_id,
                "region": args.region,
                "bucket_name": args.bucket_name,
                "pipeline_sa_email": args.pipeline_sa,
                "model_output_uri": model_output_uri,
                "training_image_uri": args.training_image,
                "serving_container_image_uri": args.serving_image,
                "model_id": args.model_id,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_csv_gcs": args.dataset_csv_gcs,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            }
        )
        
        print(f"Submitting pipeline run under service account: {args.pipeline_sa}...")
        job.submit(service_account=args.pipeline_sa)
        print("Pipeline run submitted successfully!")
        print(f"Pipeline job link: {job.resource_name}")

if __name__ == "__main__":
    main()
