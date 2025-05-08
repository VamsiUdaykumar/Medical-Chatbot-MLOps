output "controller_floating_ip" {
  value = openstack_networking_floatingip_v2.controller_fip.address
}

output "worker_ips" {
  value = [for instance in openstack_compute_instance_v2.worker : instance.access_ip_v4]
}
