variable "project_name" {}
variable "username" {}
variable "password" {}
variable "keypair_name" {}
variable "public_key_path" {}
variable "image_name" {
  default = "CC-Ubuntu24.04"
}
variable "flavor_name" {
  default = "m1.medium"
}
variable "network_name" {
  default = "sharednet1"
}
