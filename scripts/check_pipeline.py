from google.cloud import aiplatform

def check_pipeline():
    aiplatform.init(project="darenwkt-sandbox", location="us-central1")
    jobs = aiplatform.PipelineJob.list(order_by="create_time desc")
    if not jobs:
        print("No pipeline jobs found.")
        return
    for j in jobs[:5]:
        print(f"Pipeline: {j.display_name}")
        print(f"  Name/ID: {j.name}")
        print(f"  State: {j.state}")
        print(f"  Create Time: {j.create_time}")
        try:
            if hasattr(j, 'error_message') and j.error_message:
                print(f"  Error Message: {j.error_message}")
        except Exception:
            pass

if __name__ == "__main__":
    check_pipeline()
