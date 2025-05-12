variable "suffix" {
  description = "Suffix for resource names (use net ID)"
  type        = string
  default     = "project17"
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "id_rsa_new"
}

variable "nodes" {
  type = map(string)
  default = {
    "controller-project17" = "192.168.17.25"
    "worker1-project17"    = "192.168.17.26"
    "worker0-project17"    = "192.168.17.27"
  }
}
