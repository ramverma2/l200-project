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

variable "project_id" {
  description = "The Google Cloud Project ID to deploy infrastructure into."
  type        = string
  default     = "l200-final-project-tmp-argolis"
}

variable "region" {
  description = "Google Cloud Region for Vertex AI Reasoning Engine and Cloud Storage."
  type        = string
  default     = "us-east1"
}

variable "engine_display_name" {
  description = "Display name for the Vertex AI Agent Runtime reasoning engine."
  type        = string
  default     = "l200-project"
}

variable "min_instances" {
  description = "Minimum container instances for Vertex AI Agent Runtime."
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum container instances for Vertex AI Agent Runtime."
  type        = number
  default     = 2
}
