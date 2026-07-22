import argparse
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from google.cloud import aiplatform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint-display-name", default="anomaly-detection-endpoint")
    parser.add_argument("--data-path", default="data/serving_data_skewed.csv")
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to send")
    parser.add_argument("--output-dir", default="photo", help="Directory to save output plots")
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
    df_subset = df.head(args.limit).copy()
    instances = df_subset.values.tolist()

    print(f"Sending {len(instances)} prediction requests...")
    chunk_size = 50
    all_predictions = []
    
    for i in range(0, len(instances), chunk_size):
        chunk = instances[i:i + chunk_size]
        response = endpoint.predict(instances=chunk)
        # Predictions are list of floats (1.0 or -1.0)
        all_predictions.extend(response.predictions)
        time.sleep(0.2)

    df_subset['prediction'] = all_predictions

    # Separate normal and anomaly data
    normal_df = df_subset[df_subset['prediction'] == 1.0]
    anomaly_df = df_subset[df_subset['prediction'] == -1.0]

    # Plot configuration
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot normal points in green
    ax.scatter(
        normal_df['feature1'], normal_df['feature2'],
        color='#10b981', alpha=0.7, label='Normal (Predicted: 1)',
        edgecolors='none', s=35
    )

    # Plot anomaly points in red
    ax.scatter(
        anomaly_df['feature1'], anomaly_df['feature2'],
        color='#ef4444', alpha=0.9, label='Anomaly (Predicted: -1)',
        edgecolors='none', s=45
    )

    # Customize grid & spines
    ax.grid(True, linestyle='--', alpha=0.2, color='#9ca3af')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4b5563')
    ax.spines['bottom'].set_color('#4b5563')

    # Add labels, legend and title
    dataset_name = os.path.basename(args.data_path)
    ax.set_title(f'Gemini Enterprise Agent Platform Predictions for Anomaly Detection ({dataset_name})', fontsize=15, fontweight='bold', pad=20, color='#f3f4f6')
    ax.set_xlabel('Feature 1', fontsize=12, labelpad=10, color='#d1d5db')
    ax.set_ylabel('Feature 2', fontsize=12, labelpad=10, color='#d1d5db')
    ax.tick_params(colors='#9ca3af')

    legend = ax.legend(frameon=True, facecolor='#1f2937', edgecolor='#4b5563', fontsize=11, loc='upper right')
    for text in legend.get_texts():
        text.set_color('#f3f4f6')

    plt.tight_layout()

    # Save output plot
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_base = os.path.splitext(dataset_name)[0]
    save_path = os.path.join(args.output_dir, f"anomalies_plot_{dataset_base}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Successfully generated and saved plot to {save_path}")

if __name__ == "__main__":
    main()
