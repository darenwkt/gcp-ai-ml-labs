# Vertex AI Anomaly Detection Pipeline with Model Monitoring & Retraining

This project implements an end-to-end MLOps pipeline on Google Cloud using Vertex AI. It trains an **Isolation Forest** (unsupervised anomaly detection) model, deploys it to a Vertex AI Endpoint, monitors the endpoint for training-serving skew, and automatically triggers a retraining pipeline when skew is detected.

All resources are provisioned and managed via **Terraform**, fully compatible with GCP **Infrastructure Manager** and categorized with **Labels**.

## Cloud-Native Architecture

1. **Declarative Terraform**: Defines GCS buckets, the Vertex AI Endpoint, Pub/Sub alerts, Cloud Logging routers, and the retraining trigger Cloud Function. All resources are tagged with common metadata labels.
2. **Kubeflow Pipeline (`pipeline/`)**: Packages the core machine learning logic. On run, KFP:
   - Trains the Isolation Forest model on GCS data.
   - Registers the model and deploys it to the Vertex Endpoint.
   - Automatically provisions/updates the **Vertex AI Model Deployment Monitoring Job** to detect feature skew.
3. **Retraining Loop (`trigger_function/`)**:
   - Skew alerts log to Cloud Logging.
   - A Cloud Logging Sink routes these logs to a Pub/Sub topic.
   - The Pub/Sub topic triggers a Gen 2 Cloud Function.
   - The Cloud Function executes the KFP Pipeline to retrain the model on updated training data.

---

## File Structure

```text
├── data/
│   ├── training_data.csv          # Normal distribution baseline data
│   └── serving_data_skewed.csv    # Skewed data to trigger drift detection
├── pipeline/
│   ├── pipeline.py                # KFP Pipeline definition
│   └── pipeline.yaml              # Compiled pipeline spec (ready for Vertex)
├── scripts/
│   ├── generate_data.py           # Generates synthetic CSV data
│   └── predict.py                 # Client to send prediction requests
└── terraform/
    ├── main.tf                    # GCP providers, GCS buckets, API activation
    ├── variables.tf               # Terraform input variables
    ├── vertex.tf                  # Vertex AI Endpoint
    └── trigger.tf                 # Pub/Sub, Log router, Cloud Function (Gen 2)
```

---

## Getting Started

### Prerequisites

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
2. Authenticate with your Google Cloud account:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

---

## Deployment Option A: Google Cloud Infrastructure Manager (Recommended)
This method executes your Terraform configuration directly in the cloud on Google's managed Infrastructure Manager, bypassing local execution security policies (like Santa).

1. **Enable the Infrastructure Manager API** in your project:
   ```bash
   gcloud services enable config.googleapis.com
   ```
2. **Deploy the configuration** from the `terraform/` directory:
   ```bash
   cd terraform
   gcloud infra-manager deployments apply anomaly-detection-deployment \
       --local-source="." \
       --location="us-central1" \
       --input-values="project_id=YOUR_PROJECT_ID" \
       --service-account="projects/YOUR_PROJECT_ID/serviceAccounts/infra-manager-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
       --project="YOUR_PROJECT_ID"
   ```
3. You can track this deployment under **Infrastructure Manager** in your GCP Console.

---

## Deployment Option B: Local Terraform CLI
If you prefer to run Terraform on your local machine:

1. **Install Terraform** on your machine.
2. Navigate to the `terraform/` directory:
   ```bash
   cd terraform
   terraform init
   terraform apply -var="project_id=YOUR_PROJECT_ID"
   ```

---

## How to Test and Verify

Since the Endpoint is deployed empty initially, you must trigger the pipeline for the **first time** to deploy your model and configure monitoring.

### 1. Trigger the Initial Run
Publish a trigger message to the retraining Pub/Sub topic to launch the first KFP run:
```bash
gcloud pubsub topics publish anomaly-retraining-alerts --message="{\"initial-trigger\": true}"
```
*Wait for the Vertex AI Pipeline job to finish in your GCP Console. This will deploy the model and set up the monitoring job.*

### 2. Send Normal Traffic
Run the prediction client to send baseline prediction queries. This populates Vertex AI's prediction logs:
```bash
python3 scripts/predict.py --project YOUR_PROJECT_ID --data-path data/training_data.csv
```

### 3. Simulate Training-Serving Skew
Send prediction requests using the skewed serving dataset:
```bash
python3 scripts/predict.py --project YOUR_PROJECT_ID --data-path data/serving_data_skewed.csv
```

### 4. Wait for Automatic Retraining
1. Vertex AI Model Monitoring runs every hour. It analyzes the serving requests logged to BigQuery and calculates the statistical distance between serving and baseline (training) distributions.
2. If the skew exceeds the threshold (e.g. `0.01`), Vertex AI logs a skew detection alert.
3. The Cloud Logging router catches this log and sends a message to the `anomaly-retraining-alerts` Pub/Sub topic.
4. The `retrain-trigger-function` Cloud Function receives the Pub/Sub message and triggers the Vertex AI Pipeline to start retraining.


