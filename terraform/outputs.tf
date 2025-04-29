output "controller_ip" {
  value = openstack_compute_instance_v2.controller.access_ip_v4
}

output "worker_ips" {
  value = [for instance in openstack_compute_instance_v2.worker : instance.access_ip_v4]
}
