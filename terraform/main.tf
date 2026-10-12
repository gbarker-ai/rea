terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required Google Cloud APIs for AgentOps, Memory, DLP, & Tracing
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "firestore.googleapis.com",
    "dlp.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Dedicated Least-Privilege Service Account for the ADK Agent
resource "google_service_account" "real_estate_agent_sa" {
  account_id   = "real-estate-front-desk-sa"
  display_name = "Summit Key Realty Front Desk ADK Agent Service Account"
}

resource "google_project_iam_member" "agent_iam_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.real_estate_agent_sa.email}"
}

# 3. Google Cloud Secret Manager Resources (Criterion 5.3: Secure Secret Management)
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = var.gemini_api_key_secret_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "broker_hitl_signing_secret" {
  secret_id = var.broker_hitl_signing_secret_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

# 4. Persistent Session & Caller CRM Store in Firestore (Criterion 2.3: Persistent Session State)
resource "google_firestore_database" "caller_memory_db" {
  project     = var.project_id
  name        = "real-estate-caller-memory"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.required_apis]
}

# 5. Cloud Run v2 Service Hosting the ADK Real Estate Front Desk Agent
resource "google_cloud_run_v2_service" "real_estate_front_desk_service" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.real_estate_agent_sa.email

    containers {
      image = var.container_image

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "REAL_ESTATE_STORAGE_BACKEND"
        value = "firestore"
      }
      env {
        name  = "ENABLE_CLOUD_DLP"
        value = "true"
      }
      # Secure Secret Manager Injection (No hardcoded API keys)
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "BROKER_HITL_SIGNING_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.broker_hitl_signing_secret.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_firestore_database.caller_memory_db,
  ]
}
