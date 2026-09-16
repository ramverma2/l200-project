# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

output "project_id" {
  description = "The Google Cloud Project ID."
  value       = var.project_id
}

output "region" {
  description = "The deployment region."
  value       = var.region
}

output "service_account_email" {
  description = "The service account email assigned to the agent."
  value       = google_service_account.agent_sa.email
}

output "telemetry_bucket_name" {
  description = "GCS bucket used for telemetry, structured logs, and vector storage."
  value       = google_storage_bucket.agent_telemetry.name
}

output "reasoning_engine_id" {
  description = "Resource ID of the deployed Vertex AI Reasoning Engine."
  value       = google_vertex_ai_reasoning_engine.blackjack_engine.id
}
