import sys
import time
import argparse
from google.cloud import aiplatform

def monitor_pipeline(project_id: str, region: str, job_id: str):
    aiplatform.init(project=project_id, location=region)
    resource_name = f"projects/{project_id}/locations/{region}/pipelineJobs/{job_id}"
    print(f"Monitoring pipeline run: {resource_name}")
    
    terminal_states = [
        "PIPELINE_STATE_SUCCEEDED",
        "PIPELINE_STATE_FAILED",
        "PIPELINE_STATE_CANCELLED",
        "PIPELINE_STATE_PAUSED"
    ]
    
    while True:
        try:
            job = aiplatform.PipelineJob.get(resource_name)
            state = job.state.name
            print(f"Current Pipeline State: {state}")
            
            # Print tasks progress
            if job.gca_resource.job_detail and job.gca_resource.job_detail.task_details:
                print("Tasks status:")
                for detail in job.gca_resource.job_detail.task_details:
                    print(f"  - {detail.task_name}: {detail.state.name}")
            
            if state in terminal_states:
                print(f"Pipeline reached terminal state: {state}")
                if state != "PIPELINE_STATE_SUCCEEDED":
                    # Print error details
                    print(f"Pipeline error: {job.gca_resource.error}")
                    sys.exit(1)
                else:
                    print("Pipeline succeeded!")
                    sys.exit(0)
                    
        except Exception as e:
            print(f"Error querying job: {e}")
            
        time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--job-id", type=str, required=True)
    args = parser.parse_args()

    monitor_pipeline(args.project_id, args.region, args.job_id)
