import sys
import argparse
from google.cloud import aiplatform

def check_status(project_id: str, region: str, job_id: str):
    aiplatform.init(project=project_id, location=region)
    resource_name = f"projects/{project_id}/locations/{region}/pipelineJobs/{job_id}"
    print(f"Retrieving Pipeline Job {resource_name}...")
    try:
        job = aiplatform.PipelineJob.get(resource_name)
        state = job.state
        print(f"PIPELINE_STATE_STATUS:{state.name}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--job-id", type=str, required=True)
    args = parser.parse_args()

    check_status(args.project_id, args.region, args.job_id)
