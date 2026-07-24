# Google Cloud Platform (GCP) AI & Machine Learning Labs

Welcome to the **GCP AI/ML Labs** repository! This is an educational, hands-on codebase designed to help developers, data scientists, and cloud architects master production-grade Machine Learning and Artificial Intelligence on Google Cloud Platform.

Rather than just focusing on model algorithms in isolated Jupyter notebooks, these labs emphasize **production MLOps, scalability, automation, monitoring, and Infrastructure-as-Code (IaC)** using modern GCP services.

---

## 🚀 Lab Catalog

| Lab | Difficulty | Focus Area | GCP Services Used | Links |
| :--- | :--- | :--- | :--- | :--- |
| **01. Gemini Enterprise Agent Platform MLOps Pipeline** | Intermediate | Model Monitoring, Retraining, IaC | Gemini Enterprise Agent Platform, BigQuery, Cloud Functions, Pub/Sub, Terraform | [Lab Readme](01_mlops_pipeline/README.md) |
| **03. IT Support Ticket LLM Fine-Tuning & Serving** | Advanced | PEFT/LoRA Fine-tuning, Vertex AI Pipelines, serving | Vertex AI, Artifact Registry, GCS, Terraform, Infrastructure Manager | [Lab Readme](03_model_fine_tuning/README.md) |
| *More coming soon...* | - | Generative AI, Batch Inferencing, Feature Store | Gemini Enterprise Agent Platform Agent Builder, Dataflow, Gemini Enterprise Agent Platform Feature Store | *In Development* |

---

## 🛠️ Detailed Lab Summaries

### Lab 1: Anomaly Detection with Model Monitoring & Automated Retraining
Build an end-to-end unsupervised anomaly detection system using an **Isolation Forest** model. The lab covers provisioning resources, deploying to Gemini Enterprise Agent Platform, monitoring for statistical skew, and triggering automatic retraining.

```
       +-------------------------------------------------------------+
       |                     Terraform/IaC Setup                     |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                 Kubeflow Training Pipeline                  |
       |     Prepare Data -> Train model -> Deploy -> Monitor        |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |       Gemini Enterprise Agent Platform Model Endpoint       |
       +-------------------------------------------------------------+
             |                                              ^
             v (Serving Logs)                               | (Retrains)
       +----------------------------+   Logs Drift    +--------------+
       | Gemini Enterprise Agent    | --------------> |Cloud Function|
       | Platform Monitoring        |                 |  (Launches   |
       | (Analyzes BigQuery)        |                 |  Pipeline)   |
       +----------------------------+                 +--------------+
```

#### Key Concepts Learned:
* **Infrastructure as Code (IaC)**: Deploying Gemini Enterprise Agent Platform endpoints, metadata labels, and serverless pipelines declaratively with Terraform.
* **Continuous Monitoring**: Configuring skew detection using Gemini Enterprise Agent Platform Model Monitoring to detect changes in prediction distributions.
* **Serverless Event-Driven Architecture**: Routing Gemini Enterprise Agent Platform alerts through Cloud Logging and Pub/Sub to trigger a Gen 2 Cloud Function that kicks off a new training run.
* **Model Deployment Strategies**: Automating deployment splits and cleanly undeploying older model versions.

👉 **Get started with [Lab 1: 01_mlops_pipeline](01_mlops_pipeline/README.md).**

---

### Lab 3: IT Support Ticket LLM Fine-Tuning & Serving Pipeline
Fine-tune a pre-trained LLM (such as `gpt2` or `Qwen2.5-0.5B-Instruct`) on a dataset of IT support tickets using Hugging Face **PEFT (LoRA)** inside a Vertex AI Pipeline. The pipeline automates dataset preparation, single-GPU training, model merging, registration in Vertex AI Model Registry, and deployment to a serverless Vertex AI Endpoint.

```
+--------------------------+     SFTTrainer (LoRA)      +----------------------------+
| Hugging Face / GCS IT    | ------------------------> |    Fine-Tuning Component   |
|      Tickets Dataset     |                           |       (Single-GPU T4)      |
+--------------------------+                           +----------------------------+
                                                                     |
                                                                     v
+--------------------------+      Auto-Deployment      +----------------------------+
|   Vertex AI Prediction   | <------------------------ |  Vertex AI Model Registry  |
|         Endpoint         |                           |  (Merged weights uploaded) |
+--------------------------+                           +----------------------------+
```

#### Key Concepts Learned:
* **LLM Fine-Tuning & PEFT**: Implementing parameter-efficient SFT fine-tuning (LoRA) using Hugging Face transformers, PEFT, and TRL datasets.
* **Continuous Integration & Delivery (CI/CD)**: Compiling and executing Vertex AI Pipeline Runs to orchestrate model fine-tuning and endpoint deployment.
* **Serverless Serving**: Deploying a containerized Flask model server loading fine-tuned weights on a Vertex AI endpoint.

👉 **Get started with [Lab 3: 03_model_fine_tuning](03_model_fine_tuning/README.md).**


---

## 📐 General Prerequisites

Before starting any of the labs, ensure you have:

1. **A Google Cloud Platform Project** with billing enabled.
2. **Google Cloud SDK (gcloud)** installed on your machine. [Installation Guide](https://cloud.google.com/sdk/docs/install).
3. **Application Default Credentials** configured locally:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
4. **Terraform CLI** installed if you choose to deploy resources from your local command line.

---

## 💡 Learning Design Principles
Each lab follows these four core principles:
1. **Reproducible**: Every cloud resource is defined in code (Terraform).
2. **Production-First**: Focuses on monitoring, logging, and automated scaling/retraining.
3. **GCP Idiomatic**: Follows Google Cloud best practices (e.g., using GCP Infrastructure Manager, IAM least-privilege service accounts, and standard labels).
4. **Visual & Analytical**: Includes visual demonstrations of experiments, drift alerts, and model performance.
