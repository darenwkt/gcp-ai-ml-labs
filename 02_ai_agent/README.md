# Lab 02: Generative Multi-Agent Operations Troubleshooter with RAG & MCP

This project provisions an end-to-end, multi-agent operations assistant on Google Cloud using **Vertex AI Agent Builder (Dialogflow CX)** and **Discovery Engine (RAG)**. 

The system acts as a generative operations troubleshooting partner for cloud engineers, orchestrating requests to specialized domain playbooks (Compute, Storage/Database, Networking, AI/ML, Billing, and Security), querying official documentation (RAG), and calling GCP/MCP APIs for live resource states.

## Architecture Diagram

The coordinator playbook dynamically transitions execution to specialized spokes based on user requests:

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
```

---

## File Structure

```text
├── design.md                      # Detailed architectural design
├── README.md                      # Setup and deployment guide
└── terraform/
    ├── main.tf                    # GCP providers, APIs, resource settings
    ├── variables.tf               # Terraform input parameters
    ├── outputs.tf                 # Exports created resource IDs
    ├── agent.tf                   # Dialogflow CX Coordinator Agent
    ├── data_store.tf              # Discovery Engine RAG Data Store
    ├── tools.tf                   # Search & GCP API Tool Definitions
    ├── playbooks.tf               # Specialized Playbook Definitions
    └── playbooks/                 # Natural language playbook instruction markdown
        ├── coordinator.md
        ├── compute.md
        ├── db_storage.md
        ├── networking.md
        ├── aiml.md
        ├── billing.md
        └── security.md
```

---

## Deployment Instructions

### Prerequisites
1. **Google Cloud SDK**: Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) and authenticate:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
2. **Terraform**: Make sure you have Terraform (>= 1.0.0) installed.

### Setup and Deploy

1. Navigate to the `terraform/` directory:
   ```bash
   cd terraform
   ```

2. Initialize and deploy the configuration:
   ```bash
   terraform init
   terraform apply -var="project_id=YOUR_PROJECT_ID"
   ```

3. Note the outputs of the deployment:
   - **`agent_id`**: The ID of the created Dialogflow CX Coordinator Agent.
   - **`data_store_id`**: The ID of the Discovery Engine RAG Data Store.

---

## Retrieval-Augmented Generation (RAG) Data Import

By default, the Discovery Engine data store is created empty. To enable RAG:
1. Go to the **Vertex AI Agent Builder** console in your GCP console.
2. Select **Data Stores** -> **gcp-documentation-store**.
3. Click **Import Data** and select one of the following:
   - **Cloud Storage**: Provide a GCS path to uploaded GCP documentation PDFs/HTML files.
   - **Web Crawler**: Configure domain crawls for `https://cloud.google.com/docs` to automatically index documentation.

---

## Model Context Protocol (MCP) Integration

The playbooks connect to live GCP metrics and configurations via OpenAPI Tool definitions. For advanced local troubleshooting (e.g. running `kubectl` on local GKE clusters or accessing internal logs):
1. Run a secure local/private **MCP Gateway API** in your environment.
2. Register your local MCP servers (e.g. GKE, Database, logs) to the gateway.
3. Configure the webhook/OpenAPI URLs in Dialogflow CX Tools to point to your secure API gateway endpoints.
