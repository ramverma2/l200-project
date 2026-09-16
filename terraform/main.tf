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

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ─── APIs & Services ─────────────────────────────────────────────

resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# ─── Storage for Logs & Vector Store Knowledge ──────────────────

resource "google_storage_bucket" "agent_telemetry" {
  name                        = "${var.project_id}-blackjack-telemetry"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 90
    }
  }

  depends_on = [google_project_service.required_apis]
}

# ─── Service Account for Agent Runtime ──────────────────────────

resource "google_service_account" "agent_sa" {
  account_id   = "blackjack-agent-sa"
  display_name = "Blackjack Strategy Tutor Agent Runtime SA"
  project      = var.project_id
}

resource "google_project_iam_member" "agent_aiplatform" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "agent_trace" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "agent_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_storage_bucket_iam_member" "agent_storage" {
  bucket = google_storage_bucket.agent_telemetry.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.agent_sa.email}"
}

# ─── Vertex AI Agent Runtime / Reasoning Engine ─────────────────

resource "google_vertex_ai_reasoning_engine" "blackjack_engine" {
  provider     = google-beta
  project      = var.project_id
  region       = var.region
  display_name = var.engine_display_name
  description  = "Production Multi-Agent Blackjack Strategy Tutor on Agent Runtime"

  spec {
    source_code_gcs_uri = "gs://${google_storage_bucket.agent_telemetry.name}/src/agent.tar.gz"
    
    agent_engine_spec {
      model_deployment_spec {
        min_instances = var.min_instances
        max_instances = var.max_instances
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_project_iam_member.agent_aiplatform,
  ]
}
