variable "project_id" {
  description = "Google Cloud Project ID hosting the Real Estate Front Desk ADK Agent."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Vertex AI, and Firestore."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name for the Real Estate Front Desk Agent."
  type        = string
  default     = "real-estate-front-desk-agent"
}

variable "container_image" {
  description = "Artifact Registry container image URI built via ADK CLI (`adk deploy cloud_run`)."
  type        = string
  default     = "us-central1-docker.pkg.dev/real-estate-agent-prod/agents/front-desk:latest"
}

variable "gemini_api_key_secret_id" {
  description = "Secret Manager ID storing the Gemini / Vertex API credential."
  type        = string
  default     = "real-estate-agent-gemini-api-key"
}

variable "broker_hitl_signing_secret_id" {
  description = "Secret Manager ID storing the HMAC signing key for Human-in-the-Loop broker approvals."
  type        = string
  default     = "real-estate-agent-broker-hitl-secret"
}
