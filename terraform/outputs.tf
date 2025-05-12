output "node_private_ips" {
  value = {
    for k, v in openstack_networking_port_v2.private_net_ports :
    k => v.all_fixed_ips[0]
  }
}
