import argparse
import pandas as pd
import time
from google.cloud import aiplatform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint-display-name", default="anomaly-detection-endpoint")
    parser.add_argument("--data-path", default="data/serving_data_skewed.csv")
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to send")
    args = parser.parse_args()

    aiplatform.init(project=args.project, location=args.region)

    print(f"Finding endpoint '{args.endpoint_display_name}'...")
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{args.endpoint_display_name}"',
        order_by="create_time desc"
    )

    if not endpoints:
        print(f"Error: Endpoint '{args.endpoint_display_name}' not found.")
        return
    
    endpoint = endpoints[0]
    print(f"Found endpoint: {endpoint.resource_name}")

    print(f"Reading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Limit number of instances to send
    df_subset = df.head(args.limit)
    instances = df_subset.values.tolist()

    print(f"Sending {len(instances)} prediction requests to the endpoint...")
    # Send predictions in small chunks to simulate serving traffic and avoid payload limit
    chunk_size = 50
    for i in range(0, len(instances), chunk_size):
        chunk = instances[i:i + chunk_size]
        print(f"Sending chunk {i // chunk_size + 1} ({len(chunk)} instances)...")
        
        # Vertex AI Endpoint predict expects list of instances
        response = endpoint.predict(instances=chunk)
        
        # Isolation Forest outputs: 1 for normal, -1 for anomaly
        predictions = response.predictions
        print(f"Received predictions: {predictions[:10]} ...")
        
        # Sleep briefly to mimic streaming traffic
        time.sleep(0.5)

    print("Finished sending prediction requests.")

if __name__ == "__main__":
    main()
