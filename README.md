# Google Cloud Platform (GCP) AI & Machine Learning Labs

Welcome to the **GCP AI/ML Labs** repository! This is an educational, hands-on codebase designed to help developers, data scientists, and cloud architects master production-grade Machine Learning and Artificial Intelligence on Google Cloud Platform.

Rather than just focusing on model algorithms in isolated Jupyter notebooks, these labs emphasize **production MLOps, scalability, automation, monitoring, and Infrastructure-as-Code (IaC)** using modern GCP services.

---

## 🚀 Lab Catalog

| Lab | Difficulty | Focus Area | GCP Services Used | Links |
| :--- | :--- | :--- | :--- | :--- |
| **01. Gemini Enterprise Agent Platform MLOps Pipeline** | Intermediate | Model Monitoring, Retraining, IaC | Gemini Enterprise Agent Platform, BigQuery, Cloud Functions, Pub/Sub, Terraform | [Lab Readme](01_mlops_pipeline/README.md) |
| **02. Distributed GPT-2 Training and Deployment** | Advanced | 3D Parallelism, Multi-worker pretraining, SFT Fine-tuning (LoRA), Serving | Gemini Enterprise Agent Platform, Kubeflow Pipelines, Artifact Registry, Terraform | [Lab Readme](02_model_training/README.md) |

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

### Lab 2: Distributed GPT-2 pretraining (3D Parallelism) & Instruction Fine-Tuning (SFT)
Implement a manual, zero-dependency 3D Distributed Parallelism grid topology (Data Parallelism, Pipeline Parallelism, Tensor Parallelism) to train a GPT-2 (124M) language model on Google Cloud Platform, followed by Instruction SFT Fine-tuning (LoRA) and auto-deployment to a serving endpoint.

```
+--------------------------+     preprocess.py / HF     +----------------------------+
| HuggingFace FineWeb-Edu  | ------------------------> |    Pretraining CustomJob   |
|         Dataset          |                           | (DP x PP x TP on 12x T4s)  |
+--------------------------+                           +----------------------------+
                                                                     |
                                                                     v
+--------------------------+      Auto-Deployment      +----------------------------+
|  Vertex AI Prediction    | <------------------------ |   LoRA Fine-tuned Model    |
|         Endpoint         |                           |      (SFT / finetune)      |
+--------------------------+                           +----------------------------+
```

#### Key Concepts Learned:
* **3D Distributed Parallelism**: Implementing manual tensor-sharding (TP), layer-pipelining (PP), and gradient-averaging (DP) in native PyTorch.
* **Orchestrating Complex ML Workflows**: Building a multi-node Vertex AI CustomJob pretraining step inside a Kubeflow Pipeline.
* **Instruction Fine-Tuning**: Wrapping PyTorch linear projections with Low-Rank Adapters (LoRA) and merging weights back for deployment.
* **Serverless Serving**: Deploying a containerized Flask model server to a Vertex AI serving endpoint.

👉 **Get started with [Lab 2: 02_model_training](02_model_training/README.md).**


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
