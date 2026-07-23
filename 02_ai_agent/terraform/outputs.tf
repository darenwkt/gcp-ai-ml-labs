output "agent_id" {
  value       = google_dialogflow_cx_agent.coordinator.id
  description = "The ID of the created Dialogflow CX Agent."
}

output "data_store_id" {
  value       = google_discovery_engine_data_store.gcp_docs.id
  description = "The ID of the created Discovery Engine Data Store (RAG)."
}

output "search_tool_id" {
  value       = google_dialogflow_cx_tool.search_tool.id
  description = "The ID of the RAG Search Tool."
}
