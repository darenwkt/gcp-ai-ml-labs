resource "google_dialogflow_cx_tool" "search_tool" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "gcp-documentation-search"
  description  = "Search official GCP documentation for troubleshooting steps, quotas, configurations, and best practices."
  
  data_store_spec {
    data_store = google_discovery_engine_data_store.gcp_docs.id
  }
}

resource "google_dialogflow_cx_tool" "compute_tool" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "gcloud-compute-tool"
  description  = "API tool to fetch live GKE cluster and VM instance states."

  openapi_spec {
    text = jsonencode({
      openapi = "3.0.0"
      info = {
        title   = "GCP Compute Proxy API"
        version = "1.0.0"
      }
      paths = {
        "/instances" = {
          get = {
            summary     = "List VM instances and statuses"
            description = "Returns a JSON list of VM instances in the project."
            responses = {
              "200" = {
                description = "Success"
              }
            }
          }
        }
      }
    })
  }
}

resource "google_dialogflow_cx_tool" "db_storage_tool" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "gcloud-db-storage-tool"
  description  = "API tool to fetch live Database (Cloud SQL) metrics and bucket policies."

  openapi_spec {
    text = jsonencode({
      openapi = "3.0.0"
      info = {
        title   = "GCP Database and Storage Proxy API"
        version = "1.0.0"
      }
      paths = {
        "/databases" = {
          get = {
            summary     = "List database states and CPU connections"
            responses = {
              "200" = {
                description = "Success"
              }
            }
          }
        }
      }
    })
  }
}

resource "google_dialogflow_cx_tool" "network_tool" {
  provider     = google-beta
  parent       = google_dialogflow_cx_agent.coordinator.id
  display_name = "gcloud-network-tool"
  description  = "API tool to query VPC subnets and active firewall policies."

  openapi_spec {
    text = jsonencode({
      openapi = "3.0.0"
      info = {
        title   = "GCP Networking Proxy API"
        version = "1.0.0"
      }
      paths = {
        "/firewalls" = {
          get = {
            summary     = "List active project firewall rules"
            responses = {
              "200" = {
                description = "Success"
              }
            }
          }
        }
      }
    })
  }
}
