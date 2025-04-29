provider "openstack" {
  auth_url    = "https://kvm.tacc.chameleoncloud.org:5000/v3"
  tenant_name = var.project_name
  domain_name = "default"
  region      = "KVM@TACC"
  user_name   = var.username
  password    = var.password
}

resource "openstack_compute_keypair_v2" "keypair" {
  name       = var.keypair_name
  public_key = file(var.public_key_path)
}

resource "openstack_compute_instance_v2" "controller" {
  name            = "controller-node"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = ["default"]
  network {
    name = var.network_name
  }
}

resource "openstack_compute_instance_v2" "worker" {
  count           = 2
  name            = "worker-node-${count.index}"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = ["default"]
  network {
    name = var.network_name
  }
}
