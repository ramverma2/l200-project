variable "project_id" {
  description = "The Google Cloud Project ID."
  type        = string
  default     = "l200-final-project-tmp-argolis"
}

variable "region" {
  description = "Google Cloud Region."
  type        = string
  default     = "us-east1"
}

variable "engine_display_name" {
  description = "Display name for the Vertex AI Agent Runtime reasoning engine."
  type        = string
  default     = "l200-project"
}

variable "min_instances" {
  description = "Minimum container instances."
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum container instances."
  type        = number
  default     = 2
}
