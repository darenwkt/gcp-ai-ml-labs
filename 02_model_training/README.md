# GCP GPT-2 Training and Deployment Pipeline

This repository implements a complete end-to-end pipeline to train, deploy, and serve a GPT-2 (124M) language model on Google Cloud Platform (GCP) using **Gemini Enterprise Agent Platform** and **Kubeflow Pipelines (KFP)**.

> [!NOTE]
> * **Interactive Model Architecture Visualizer:** [LLM Visualizer - Model Architecture](https://darenwkt.github.io/llm-visualizer/#/model-architecture)
> * **Interactive 3D Distributed Training Visualizer:** [LLM Visualizer - Distributed Training](https://darenwkt.github.io/llm-visualizer/#/distributed-training)

---

## 📖 Table of Contents
1. [Project Structure](#-project-structure)
2. [🚀 Quick Start Guide](#-quick-start-guide)
3. [🧠 Pretraining vs. Instruction Fine-Tuning (SFT)](#-pretraining-vs-instruction-fine-tuning-sft)
4. [📊 Hardware and Performance Benchmarks](#-hardware-and-performance-benchmarks)
5. [🛡️ Detailed Guide: 3D Distributed Parallelism](#-detailed-guide-3d-distributed-parallelism)
    * [3D Process Grid Topology](#1-3d-process-grid-topology)
    * [Tensor Parallelism (TP)](#2-tensor-parallelism-tp)
    * [Pipeline Parallelism (PP)](#3-pipeline-parallelism-pp)
    * [Data Parallelism (DP) & Manual Gradient Sync](#4-data-parallelism-dp--manual-gradient-sync)
    * [Checkpoint Consolidation](#5-checkpoint-consolidation)

---

## 📁 Project Structure

```
├── .gitignore
├── README.md               # Project documentation
├── pipeline/
│   ├── pipeline.py         # KFP pipeline definition (dynamic resource constraints)
│   ├── pipeline.yaml       # Compiled pipeline specification
│   └── run_pipeline.py     # Command-line runner to compile and submit pipeline runs
├── scripts/
│   ├── preprocess.py       # Pre-tokenizes Hugging Face dataset (FineWeb-Edu) to GCS
│   ├── predict_local.py    # Local model loading and text generation test script
│   └── test_endpoint.py    # Tests predictions against a deployed Gemini Enterprise Agent Platform Endpoint
├── src/
│   ├── training/
│   │   ├── train.py        # GPT-2 model definition and PyTorch pretraining loop
│   │   ├── Dockerfile      # Container definition for Gemini Enterprise Agent Platform training jobs
│   │   └── requirements.txt
│   └── prediction/
│       ├── main.py         # Flask-based serving app for prediction container
│       ├── Dockerfile      # Container definition for Gemini Enterprise Agent Platform online prediction
│       └── requirements.txt
└── terraform/              # Terraform templates for GCP resource provisioning
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have the Google Cloud SDK and Python virtual environment configured:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/training/requirements.txt
```

> [!IMPORTANT]
> **Configuration & Environment Setup**:
> To prevent hardcoded project IDs and GCS paths, the Python scripts dynamically read configurations from environment variables. Before executing pipeline and deployment scripts, make sure to set the following variables in your terminal:
> ```bash
> # Set your active Google Cloud Project ID
> export GOOGLE_CLOUD_PROJECT="<your-gcp-project-id>"
> 
> # (Optional) Override custom GCS output path and container image URI
> export MODEL_GCS_URI="gs://<your-gcs-bucket>/model-output/<timestamp>"
> export SERVING_IMAGE_URI="us-central1-docker.pkg.dev/<your-gcp-project-id>/gpt2-prediction-images/gpt2-predict:latest"
> ```


### 2. Infrastructure Setup (Terraform)
Provision the required GCP resources (Storage Bucket, Artifact Registry repository, Service Accounts, and IAM bindings):
```bash
cd terraform
terraform init
terraform apply -var="project_id=<your-project-id>"
cd ..
```
*Outputs generated:*
* **GCS Storage Bucket**: `<project-id>-gpt2-pipeline-artifacts`
* **Artifact Registry URL**: `us-central1-docker.pkg.dev/<project-id>/gpt2-prediction-images`
* **Pipeline Service Account**: `gpt2-pipeline-sa@<project-id>.iam.gserviceaccount.com`

### 3. Dataset Preprocessing
Tokenize the FineWeb-Edu dataset (10B token subset) and stream it directly to GCS:
```bash
python scripts/preprocess.py \
  --subset sample-10BT \
  --num-docs -1 \
  --output-gcs-uri gs://<your-bucket-name>/dataset/train.bin
```

### 4. Build and Push Containers
Compile and push Docker images to Artifact Registry:
```bash
# Training Container
gcloud builds submit --tag us-central1-docker.pkg.dev/<project-id>/gpt2-prediction-images/gpt2-train:latest src/training

# Prediction Container
gcloud builds submit --tag us-central1-docker.pkg.dev/<project-id>/gpt2-prediction-images/gpt2-predict:latest src/prediction
```

### 5. Run the Training and Deployment Pipeline
Submit a pipeline run (e.g., for a 12x T4 cluster with 3D Parallelism sizes: `dp=2`, `tp=3`, `pp=2`):
```bash
python pipeline/run_pipeline.py \
  --project-id <project-id> \
  --bucket-name <bucket-name> \
  --serving-image us-central1-docker.pkg.dev/<project-id>/gpt2-prediction-images/gpt2-predict:latest \
  --pipeline-sa gpt2-pipeline-sa@<project-id>.iam.gserviceaccount.com \
  --gpu-type NVIDIA_TESLA_T4 \
  --gpu-limit 12 \
  --tp-size 3 \
  --pp-size 2 \
  --batch-size 8 \
  --max-steps 1000 \
  --checkpoint-activations False
```

### 6. Test Predictions
Once KFP has deployed the model endpoint, test inference:
```bash
python scripts/test_endpoint.py
```

---

## 🧠 Pretraining vs. Instruction Fine-Tuning (SFT)

The training pipeline implements a two-stage training paradigm to build a functional, instruction-following AI assistant:

```
[ Raw Web Text Corpus ] ──> ( 1. Pretraining ) ──> [ Pretrained GPT-2 (Completes Text) ]
                                                            │
[ Stanford Alpaca JSON ] ──> ( 2. SFT via LoRA ) ◄──────────┘
                                 │
                                 ▼
                     [ Conversational Assistant ]
```

<details>
<summary>🔍 Click to expand explanation of Pretraining vs SFT details</summary>

### 1. Pretraining Stage (Unsupervised Learning)
* **Goal**: Teach the model the structures of language, grammar, reasoning, and world facts.
* **Method**: The model is trained on a massive corpus of raw web text (`FineWeb-Edu`) to solve the next-token prediction task.
* **Output**: A powerful "document completer". For example, if prompted with *"Write a list of primary colors"*, it might continue with *"...and secondary colors. In this chapter we will discuss..."* instead of actually answering the question, because it acts as a generic text continuation engine.

### 2. Supervised Fine-Tuning (SFT) Stage (Instruction Tuning)
* **Goal**: Align the model's output formatting so it behaves like a helpful conversational assistant.
* **Method**: The model is trained on structured prompt-response pairs (`Instruction-Input-Output` pairs from the Alpaca dataset) using supervised target masks. The loss is computed only on response tokens, ignoring prompt inputs.
* **Output**: An AI assistant that knows how to follow user instructions, answer questions, and respond in natural conversation.

### 3. Parameter-Efficient Tuning via LoRA (Low-Rank Adaptation)
To avoid the massive hardware cost of updating all 124M weights during the fine-tuning stage:
* We freeze the pretrained base GPT-2 parameters.
* We inject trainable low-rank decomposition matrices ($W_0 + \Delta W$, where $\Delta W = B \times A$) into the query, key, value, and projection layers of the self-attention blocks.
* This updates only a tiny fraction of trainable parameters, accelerating training times, saving GPU memory, and preventing the model from forgetting its pretrained knowledge base ("catastrophic forgetting").

</details>

---

## 📊 Hardware and Performance Benchmarks

All training runs are optimized with eager mode execution and FP16 mixed precision. Below are the verified metrics for the model training loop:

### 1. GPU Throughput Comparisons
| GPU Type | Batch Size | Step Duration (ms) | Throughput (Tokens/sec) | Performance (TFLOPS) |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA A100 (40GB)** | 32 | ~293ms | **1.12M tokens/s** | **109.6 TFLOPS** |
| **NVIDIA L4 (24GB)** | 12 | ~340ms (Est.) | **362K tokens/s** | **~35.0 TFLOPS** (Est.) |
| **NVIDIA Tesla T4 (15GB)** | 8 | ~670ms (Est.) | **122K tokens/s** | **~12.0 TFLOPS** (Est.) |

### 2. Normalized Training Duration (Target: 655M Tokens / 20k steps on A100)
*Because GPU memory capacities require smaller batch sizes on L4/T4, durations are normalized to process a fixed target of **655.36 Million tokens**:*

| GPU Type | Batch Size | Required Steps | Throughput | Est. Total Duration | Scaling Factor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA A100 (40GB)** | 32 | 20,000 steps | 1.12M tokens/s | **~1h 37m** (Verified) | **1.0x** (Baseline) |
| **NVIDIA L4 (24GB)** | 12 | 53,333 steps | 362K tokens/s | **~5h 02m** (Est.) | **~3.1x slower** |
| **NVIDIA Tesla T4 (15GB)** | 8 | 80,000 steps | 122K tokens/s | **~14h 53m** (Est.) | **~9.2x slower** |

---

## 🛡️ Detailed Guide: 3D Distributed Parallelism

This repository includes a custom, zero-dependency manual implementation of **3D Parallelism** in PyTorch for scaling GPT-2 model training across massive GPU clusters (e.g. multi-node configurations) using PyTorch's native `torch.distributed` communications backend.

---

### 1. 3D Process Grid Topology

When launching training using `torchrun`, a 3D topology coordinate `(DP, PP, TP)` is calculated for every process rank:
$$\text{World Size} = \text{DP\_SIZE} \times \text{PP\_SIZE} \times \text{TP\_SIZE}$$

Each process rank is mapped to a specific coordinate:
* **Tensor Parallel (TP) Group**: Cooperating processes that shard attention heads and MLP projection matrices.
* **Data Parallel (DP) Group**: Cooperating processes that share the same layer and tensor shard, training on separate batches of text and averaging gradients.
* **Pipeline Parallel (PP) Stage**: Processes that handle sequential blocks of the model layers and communicate activations forward and gradients backward.

```
+-------------------------------------------------------------+
|                  REPLICA 0 (DP_COORD = 0)                   |
|                                                             |
|      Stage 0 (PP_COORD = 0)         Stage 1 (PP_COORD = 1)  |
|      ┌────────────────────┐         ┌────────────────────┐  |
|      │ GPU 0 (TP_COORD=0) │ ──fwd─► │ GPU 2 (TP_COORD=0) │  |
|      ├────────────────────┤         ├────────────────────┤  |
|      │ GPU 1 (TP_COORD=1) │ ──fwd─► │ GPU 3 (TP_COORD=1) │  |
|      └────────────────────┘         └────────────────────┘  |
+-------------------------------------------------------------+
          ▲                              ▲
          │ (Custom DP Gradient Sync)    │ (Custom DP Gradient Sync)
          ▼                              ▼
+-------------------------------------------------------------+
|                  REPLICA 1 (DP_COORD = 1)                   |
|                                                             |
|      Stage 0 (PP_COORD = 0)         Stage 1 (PP_COORD = 1)  |
|      ┌────────────────────┐         ┌────────────────────┐  |
|      │ GPU 4 (TP_COORD=0) │ ──fwd─► │ GPU 6 (TP_COORD=0) │  |
|      ├────────────────────┤         ├────────────────────┤  |
|      │ GPU 5 (TP_COORD=1) │ ──fwd─► │ GPU 7 (TP_COORD=1) │  |
|      └────────────────────┘         └────────────────────┘  |
+-------------------------------------------------------------+
```

<details>
<summary>⚙️ Click to expand Process Grid Limits and Constraints</summary>

For the standard GPT-2 (124M) architecture, configuration sizes are constrained by the model's dimensions:

1. **Tensor Parallel (`--tp-size`) Limits**:
   * **Min**: `1` (No sharding)
   * **Max**: `12`
   * **Constraint**: The number of attention heads (`n_heads = 12`) must be divisible by `tp_size`. Thus, valid TP sizes are: `1`, `2`, `3`, `4`, `6`, and `12`.
2. **Pipeline Parallel (`--pp-size`) Limits**:
   * **Min**: `1` (No staging)
   * **Max**: `12`
   * **Constraint**: The total blocks (`n_layers = 12`) must be divisible by `pp_size`. Thus, valid PP sizes are: `1`, `2`, `3`, `4`, `6`, and `12`.
3. **Data Parallel (`dp_size`) Limits**:
   * **Min**: `1`
   * **Max**: **Unlimited** (constrained only by total cluster GPUs). Calculated as:
     $$\text{DP\_SIZE} = \frac{\text{World Size}}{\text{TP\_SIZE} \times \text{PP\_SIZE}}$$

#### Valid 8-GPU Cluster Configuration Examples:
* **Hybrid 3D Parallelism (Default 8x T4)**: `tp_size=2`, `pp_size=2`, `dp_size=2`
* **Heavy Pipeline**: `tp_size=2`, `pp_size=4`, `dp_size=1`
* **Heavy Sharding**: `tp_size=4`, `pp_size=2`, `dp_size=1`

</details>

---

### 2. Tensor Parallelism (TP)

Attention and Feedforward layers are parallelized *within* a single transformer block by sharding projection matrices.

```
Column-Parallel Projection (e.g. QKV, MLP-fc1):
  Input X             Weight W          Column Slice        Output Y
  (B*T) x Din          Din x Dout        (Local Slice)       (Local Slice)
┌─────────┐         ┌────┬────┐         ┌────┬────┐        ┌────┬────┐
│         │         │    │    │         │ W1 │ W2 │        │ Y1 │ Y2 │
│  (B*T)  │    x    │ W1 │ W2 │   ==►   ├────┼────┤   =    │    │    │
│  x Din  │         │    │    │         │Rank│Rank│        │    │    │
└─────────┘         └────┴────┘         │ 0  │ 1  │        └──────────┘
                                        └────┴────┘      (Local Slice - No Network Sync)

Row-Parallel Projection (e.g. out_proj, MLP-fc2):
    Input X            Weight W          Row Slice            Output Y
┌────┬────┐          ┌─────────┐        ┌─────────┐         ┌─────────┐
│    │    │          │   W1    │        │  Rank 0 │         │  Consol.│
│ X1 │ X2 │    x     ├─────────┤  ==►   ├─────────┤   =     │ Output  │
│    │    │          │   W2    │        │  Rank 1 │         │  (B*T)  │
└────┴────┘          └─────────┘        └─────────┘         │  x Dout │
(Sharded input)                                             └─────────┘
                                                        (dist.all_reduce adds X1W1 + X2W2)
```

<details>
<summary>🛠️ Click to expand detailed math & logic of TP Blocks</summary>

#### A. Column-Parallel Linear (`ColumnParallelLinear`)
Splits the output features (columns) of the weight matrix across the TP group:
$$W = \begin{bmatrix} W_1 & W_2 \end{bmatrix}$$
* **Execution**: Input $X$ is multiplied locally on each rank.
* **Communication**: **Zero**. No network communication is needed during the forward pass.
* **Usage**: Query, Key, and Value projections (`W_query`, `W_key`, `W_value`) and FFN expansion layer (`fc1`).

#### B. Row-Parallel Linear (`RowParallelLinear`)
Splits the input features (rows) of the weight matrix across the TP group:
$$W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$$
* **Execution**: Input $X$ is sharded along columns $X = \begin{bmatrix} X_1 & X_2 \end{bmatrix}$. Each rank computes $Y_i = X_i \cdot W_i$ locally.
* **Communication**: **`dist.all_reduce(op=SUM)`**. To compute the true projection output $Y = X_1W_1 + X_2W_2$, outputs are summed across the TP group.
* **Usage**: Output projection (`out_proj`) and FFN contraction layer (`fc2`).

#### Transformer Block Layout
By stacking Column-Parallel and Row-Parallel layers back-to-back, we execute the entire transformer block with only **2 all-reduces**:
1. **Self-Attention**: Column-parallel QKV projections $\to$ row-parallel `out_proj` (**1 all-reduce**).
2. **FeedForward**: Column-parallel `fc1` $\to$ GELU $\to$ row-parallel `fc2` (**1 all-reduce**).

#### Why QKV is Column-Parallel and out_proj is Row-Parallel
This layout allows computing multi-head attention with **zero internal communication**:
1. **Local Head Computation**:
   By sharding $W_q, W_k, W_v$ along columns, each rank receives a disjoint subset of the attention heads (e.g. Rank 0 calculates heads 1–6, Rank 1 calculates heads 7–12).
2. **Local Softmax**:
   Since attention is computed head-by-head, dot-product calculations ($QK^T$) and non-linear Softmax are run 100% locally on each rank. If we split along rows instead, we would need multiple network syncs *inside* the attention block just to calculate the Softmax denominator.
3. **Deferred Sync**:
   The output of local attention is a column-sharded context matrix: $C = [C_1 \mid C_2]$. We feed this directly into a Row-Parallel `out_proj` layer, which performs a single `dist.all_reduce(op=SUM)` at the very end to sum the projected outputs back to the original embedding size.

```
[Self-Attention Block Flow (TP_SIZE = 2)]

                    Input X (B x T x 768)   <--- Un-sharded, duplicated on both ranks
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              [ Rank 0 ]        [ Rank 1 ]
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. QKV Projections (Column-Parallel)                            │
  │    Wq, Wk, Wv slices (768 x 384)   Wq, Wk, Wv slices (768 x 384)│
  │    Q1, K1, V1: (B x T x 384)       Q2, K2, V2: (B x T x 384)    │
  │    - Compute locally, no communication                          │
  ├─────────────────────────────────────────────────────────────────┤
  │ 2. Local Self-Attention Calculation                             │
  │    C1 = Softmax(Q1·K1^T)·V1        C2 = Softmax(Q2·K2^T)·V2     │
  │    C1 shape: (B x T x 384)         C2 shape: (B x T x 384)      │
  │    - Compute locally, no communication                          │
  ├─────────────────────────────────────────────────────────────────┤
  │ 3. Output Projection (Row-Parallel)                             │
  │    W_out1 slice: (384 x 768)       W_out2 slice: (384 x 768)    │
  │    Y1 = C1 · W_out1                Y2 = C2 · W_out2             │
  │    Y1 shape: (B x T x 768)         Y2 shape: (B x T x 768)      │
  │    - Compute locally, then sum outputs globally                 │
  └───────────┬─────────────────────────┬───────────────────────────┘
              │                         │
              ▼                         ▼
        ( Local Y1: BxTx768 )     ( Local Y2: BxTx768 )
              │                         │
              └────────► All-Reduce ◄───┘ (dist.all_reduce(op=SUM))
                            │
                            ▼
                   Output Y (B x T x 768) <--- Consolidated output on both ranks
```

#### Why FFN fc1 is Column-Parallel and fc2 is Row-Parallel
The FeedForward Network (FFN) uses the exact same stack pattern to compute local activations with zero internal communication:

```
[FeedForward (FFN) Block Flow (TP_SIZE = 2)]

                    Input X (B x T x 768)   <--- Un-sharded, duplicated on both ranks
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              [ Rank 0 ]        [ Rank 1 ]
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. FFN Expansion Layer (Column-Parallel)                        │
  │    W_fc1_1 slice: (768 x 1536)     W_fc1_2 slice: (768 x 1536)  │
  │    E1 = X · W_fc1_1                E2 = X · W_fc1_2             │
  │    E1 shape: (B x T x 1536)        E2 shape: (B x T x 1536)     │
  │    - Compute locally, no communication                          │
  ├─────────────────────────────────────────────────────────────────┤
  │ 2. Local Non-Linear Activation                                  │
  │    A1 = GELU(E1)                   A2 = GELU(E2)                │
  │    A1 shape: (B x T x 1536)        A2 shape: (B x T x 1536)     │
  │    - Compute locally, no communication                          │
  ├─────────────────────────────────────────────────────────────────┤
  │ 3. FFN Contraction Layer (Row-Parallel)                         │
  │    W_fc2_1 slice: (1536 x 768)     W_fc2_2 slice: (1536 x 768)  │
  │    Y1 = A1 · W_fc2_1               Y2 = A2 · W_fc2_2            │
  │    Y1 shape: (B x T x 768)         Y2 shape: (B x T x 768)      │
  │    - Compute locally, then sum outputs globally                 │
  └───────────┬─────────────────────────┬───────────────────────────┘
              │                         │
              ▼                         ▼
        ( Local Y1: BxTx768 )     ( Local Y2: BxTx768 )
              │                         │
              └────────► All-Reduce ◄───┘ (dist.all_reduce(op=SUM))
                            │
                            ▼
                   Output Y (B x T x 768) <--- Consolidated output on both ranks
```

</details>

---

### 3. Pipeline Parallelism (PP)

The 12 transformer blocks of GPT-2 are divided evenly across pipeline stages. For example, if `PP_SIZE = 2`:
* **Stage 0** holds **Layers 0 to 5** plus token and positional embedding tables (`tok_emb`, `pos_emb`).
* **Stage 1** holds **Layers 6 to 11** plus final layer norm (`final_norm`) and classifier head (`out_head`).

<details>
<summary>🌐 Click to expand Pipeline & Tensor Interoperability flows</summary>

In our 3D grid, pipeline stages run sequentially while tensor sharding groups run in parallel inside each stage. When Stage 0 finishes executing its local sharded blocks, it passes the sharded activation states directly to the corresponding tensor lane in Stage 1:

```
       Dataloader         Stage 0 (PP_0)             Stage 1 (PP_1)
           │               (Rank 0 & 1)               (Rank 2 & 3)
           │                    │                          │
           │────── Token ──────►│                          │
           │       Batch        │                          │
           │                    │─► Local Embeddings       │
           │                    │─► ColumnParallel (QKV)   │
           │                    │─► RowParallel (out_proj) │
           │                    │    (TP Sync Rank 0<->1)  │
           │                    │                          │
           │                    │───── Activations X ─────►│
           │                    │      (dist.send/recv)    │
           │                    │                          │
           │                    │                          │─► ColumnParallel (fc1)
           │                    │                          │─► RowParallel (fc2)
           │                    │                          │    (TP Sync Rank 2<->3)
           │                    │                          │─► Loss & Backward
           │                    │                          │
           │                    │◄────── Gradient ─────────│
           │                    │      (dist.send/recv)    │
           │                    │                          │
           │                    │─► Local Backward         │
           │                    │─► DP Grads sync          │
           │                    │   (Sync Rank 0<->4)      │
           │                    ▼                          ▼
```

During a training step, intermediate activations and gradients flow sequentially across nodes:
1. **Forward Pass**:
   * Stage 0 reads a token batch, computes embeddings and layers 0–5, and calls `dist.send(activations, dst=next_rank)`.
   * Stage 1 receives activations via `dist.recv()`, runs layers 6–11, calculates cross-entropy loss, and runs local backpropagation.
2. **Backward Pass**:
   * Stage 1 computes gradients and sends them back via `dist.send(gradients, dst=prev_rank)`.
   * Stage 0 receives gradients via `dist.recv()` and runs backward propagation on layers 5–0.

</details>

---

### 4. Data Parallelism (DP) & Manual Gradient Sync

Because our model parameters are sharded across stages and attention slices, we bypass standard PyTorch `DistributedDataParallel` (DDP) to avoid runtime crashes. 

* **Weight Initialization**: At startup, parameters are synchronized across the data parallel replicas using `dist.broadcast()` from the DP master rank of each sub-group (`dp_coord = 0`).
* **Gradient Sync**: In the custom `train_step_3d` loop, after backward propagation completes, we manually average the parameter gradients across the DP replica group via `dist.all_reduce()` before performing `optimizer.step()`.

---

### 5. Checkpoint Consolidation

During training, each pipeline stage/TP lane saves its own weight shard (`model_pp{pp}_tp{tp}.pth`) to avoid communication bottlenecks.

At the end of training, the primary rank (`loss_master`) runs the consolidation utility:
1. Downloads all sharded files from GCS/local.
2. Concatenates column-parallel layers along dimension 0.
3. Concatenates row-parallel layers along dimension 1.
4. Stitches together sequential layer blocks from each pipeline stage.
5. Outputs a standard, prediction-friendly `model.pth` that can be loaded out-of-the-box by the online prediction server.
