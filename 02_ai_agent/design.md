# Multi-Agent Cloud Troubleshooting Platform: Architecture & Design

This document outlines the architecture and design of the generative AI multi-agent system built on **Vertex AI Agent Builder (Dialogflow CX)**. The system is designed to help a cloud engineer monitor, audit, and troubleshoot resources in a Google Cloud project.

## System Architecture

The platform uses a **Hub-and-Spoke Orchestration** model:
1. **Coordinator Agent**: The user-facing entry point. It parses the engineer's query, maintains session context, and routes control dynamically to the appropriate specialized Playbook.
2. **Specialized Spokes (Playbooks)**: Goal-oriented agent playbooks that specialize in specific GCP domains.
3. **Common Tools & Integrations**:
   - **RAG Data Store**: A Google Discovery Engine data store populated with official GCP documentation to handle reference queries.
   - **MCP / GCP Tools**: Dialogflow CX Tool definitions that connect to GCP APIs or external Model Context Protocol (MCP) servers to retrieve live telemetry, logs, or resource states.

```mermaid
graph TD
    User([Cloud Engineer]) --> Coordinator[Coordinator Playbook]
    
    subgraph Specialized Agent Playbooks (Spokes)
        Coordinator --> Compute[Compute Playbook]
        Coordinator --> DB[Storage & DB Playbook]
        Coordinator --> Net[Networking Playbook]
        Coordinator --> AI[AI & ML Playbook]
        Coordinator --> Cost[Billing & Cost Playbook]
        Coordinator --> Sec[Security Playbook]
    end

    subgraph Data & Tooling Integrations
        Compute --> GCP_APIs[GCP API Tools]
        DB --> GCP_APIs
        Net --> GCP_APIs
        AI --> GCP_APIs
        Cost --> GCP_APIs
        Sec --> GCP_APIs
        
        Compute --> MCP_Server[MCP Server Tools]
        Net --> MCP_Server
        
        Compute --> RAG[(Discovery Engine RAG Data Store)]
        DB --> RAG
        Net --> RAG
        AI --> RAG
        Cost --> RAG
        Sec --> RAG
    end

    classDef playbook fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#f3f4f6;
    classDef tool fill:#1e293b,stroke:#10b981,stroke-width:2px,color:#f3f4f6;
    class Coordinator,Compute,DB,Net,AI,Cost,Sec playbook;
    class GCP_APIs,MCP_Server,RAG tool;
```

---

## Agent Playbook Designs

### 1. Coordinator Agent (The Hub)
- **Goal**: Identify the category of the engineer's problem and route them to the correct specialist playbook.
- **System Instructions**: 
  - Greet the user and ask how you can assist with their GCP environment.
  - If the request is generic, ask clarifying questions.
  - If the request is domain-specific (e.g. "Why is my database slow?"), transition control to the relevant Spoke Playbook.
  - When the Spoke Playbook completes its task or needs to hand back control, resume coordination.

### 2. Compute Specialist Playbook
- **Scope**: Compute Engine VMs, Google Kubernetes Engine (GKE) clusters, node pools, Pods, and autoscaling.
- **Capabilities**:
  - Check VM instance states (RUNNING, TERMINATED).
  - Inspect GKE cluster health, verify Pod resource allocations, and view container logs.
  - Suggest troubleshooting guides for VM ssh failures or GKE CrashLoopBackOff states.
- **Tools**: `gcloud-compute-tool`, `kubectl-mcp-tool`, `gcp-documentation-search`.

### 3. Storage & Database Specialist Playbook
- **Scope**: Cloud Storage (GCS) buckets, Cloud SQL instances, Cloud Spanner, Cloud Bigtable.
- **Capabilities**:
  - Check database CPU/memory utilization and active connections.
  - Audit GCS bucket configurations (IAM, public access, lifecycle rules).
  - Explain database performance bottlenecks or storage quota errors.
- **Tools**: `gcloud-sql-tool`, `gcloud-gcs-tool`, `gcp-documentation-search`.

### 4. Networking Specialist Playbook
- **Scope**: VPC networks, firewalls, load balancers, Cloud DNS, Cloud VPN, interconnects.
- **Capabilities**:
  - Verify firewall rules blocking traffic to a specific port.
  - Trace route/connectivity issues between two subnets.
  - Check load balancer health checks and backend status.
- **Tools**: `gcloud-network-tool`, `gcp-documentation-search`.

### 5. AI & ML Specialist Playbook
- **Scope**: Vertex AI Pipelines, Model Registries, Model Endpoints, generative model parameters.
- **Capabilities**:
  - Fetch the status of the latest retraining pipeline run.
  - Check Vertex AI endpoint latency, error rates, and active deployed models.
  - Verify Model Deployment Monitoring Job drift/skew alert statuses.
- **Tools**: `vertex-ai-tool`, `gcp-documentation-search`.

### 6. Billing & Cost Specialist Playbook
- **Scope**: Billing accounts, budgets, cost anomalies, usage metrics, forecasting.
- **Capabilities**:
  - List active budgets and check if any thresholds are breached.
  - Identify sudden spikes in daily project spend.
  - Recommend cost optimization actions (e.g. idle VM recommendations).
- **Tools**: `gcp-billing-tool`, `recommender-tool`.

### 7. Security Specialist Playbook
- **Scope**: IAM policies, Service Accounts, Cloud Logging audit logs, Security Command Center (SCC) findings.
- **Capabilities**:
  - Audit who has a specific role (e.g. Owner, Admin).
  - Query Security Command Center for high-severity vulnerability findings.
  - Trace access denied errors back to missing IAM permissions.
- **Tools**: `gcloud-iam-tool`, `scc-tool`, `gcp-documentation-search`.

---

## Retrieval-Augmented Generation (RAG) Setup

To allow the agents to answer reference questions about GCP services (e.g., "What is the recommended machine type for a production Spanner database?"), we integrate a **Google Cloud Discovery Engine Data Store**:
- **Source**: Loaded with crawling rules for `https://cloud.google.com/docs` or loaded with pre-packaged documentation PDFs/HTML files in a GCS bucket.
- **Integration**: Exposed to Dialogflow CX as a **Search Tool**. When a playbook encounters a query it cannot answer with live resource data, it queries the search tool, retrieves relevant passages, and synthesizes a grounded answer.

---

## Model Context Protocol (MCP) & Tools Integration

To allow the agents to inspect and troubleshoot live systems, they leverage **Tools** defined via OpenAPI specs:
1. **Direct GCP APIs**: Playbooks execute authenticated GCP API calls (e.g., listing instances via Compute Engine API).
2. **MCP Servers**: For complex tools (like executing `kubectl` commands, running custom database queries, or fetching details from external systems), the agent calls an API gateway that routes requests to local/secure **MCP Servers**.
   - For example, a GKE-focused MCP server can safely run `kubectl get pods --namespace=default` using local cluster credentials and return structured JSON logs to the agent.
