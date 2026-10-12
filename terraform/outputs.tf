output "cloud_run_service_url" {
  description = "Public HTTPS endpoint of the deployed Real Estate Front Desk ADK Agent."
  value       = google_cloud_run_v2_service.real_estate_front_desk_service.uri
}

output "agent_service_account_email" {
  description = "Least-privilege IAM Service Account email used by the ADK Agent."
  value       = google_service_account.real_estate_agent_sa.email
}

output "firestore_database_name" {
  description = "Firestore Native database name storing persistent caller sessions."
  value       = google_firestore_database.caller_memory_db.name
}
