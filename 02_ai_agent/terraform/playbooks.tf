resource "google_dialogflow_cx_playbook" "compute" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "Compute Specialist"
  
  goal        = "Diagnose and resolve VM and GKE cluster failures."
  instruction = file("${path.module}/playbooks/compute.md")
  
  referenced_tools = [
    google_dialogflow_cx_tool.search_tool.id,
    google_dialogflow_cx_tool.compute_tool.id
  ]
}

resource "google_dialogflow_cx_playbook" "db_storage" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "Storage & Database Specialist"
  
  goal        = "Diagnose and resolve Cloud SQL, Spanner, and GCS bucket access or latency issues."
  instruction = file("${path.module}/playbooks/db_storage.md")
  
  referenced_tools = [
    google_dialogflow_cx_tool.search_tool.id,
    google_dialogflow_cx_tool.db_storage_tool.id
  ]
}

resource "google_dialogflow_cx_playbook" "networking" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "Networking Specialist"
  
  goal        = "Diagnose firewall blockages, routing errors, and load balancer backend failures."
  instruction = file("${path.module}/playbooks/networking.md")
  
  referenced_tools = [
    google_dialogflow_cx_tool.search_tool.id,
    google_dialogflow_cx_tool.network_tool.id
  ]
}

resource "google_dialogflow_cx_playbook" "aiml" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "AI & ML Specialist"
  
  goal        = "Inspect retraining pipeline states and Vertex model monitoring skew jobs."
  instruction = file("${path.module}/playbooks/aiml.md")
  
  referenced_tools = [
    google_dialogflow_cx_tool.search_tool.id
  ]
}

resource "google_dialogflow_cx_playbook" "billing" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "Billing & Cost Specialist"
  
  goal        = "Audit project budgets, cost spikes, and recommendations."
  instruction = file("${path.module}/playbooks/billing.md")
  
  referenced_tools = [
    google_dialogflow_cx_tool.search_tool.id
  ]
}

resource "google_dialogflow_cx_playbook" "security" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "Security Specialist"
  
  goal        = "Audit IAM policies, check missing roles, and list SCC findings."
  instruction = file("${path.module}/playbooks/security.md")
  
  referenced_tools = [
    google_dialogflow_cx_tool.search_tool.id
  ]
}
