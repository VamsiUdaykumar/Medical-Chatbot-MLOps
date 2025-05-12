
# Dr. Dialog: MLOps Pipeline on Chameleon Cloud

This repository contains the full MLOps deployment of **Dr. Dialog**, an evidence-based healthcare chatbot system. The system is built and deployed following the ML SysOps methodology using Chameleon Cloud infrastructure and a Continuous Delivery pipeline with ArgoCD and Kubernetes.

## Repository Structure

```
Vamsi/
├── provision.ipynb              # Terraform-based provisioning notebook
├── deploy.sh                    # Script to orchestrate full deployment
├── dr-dialog-app.yaml           # Kubernetes deployment manifest
├── service.yaml                 # Kubernetes service manifest
├── ansible/
│   ├── ansible.cfg
│   ├── inventory.yml            # Ansible inventory for playbooks
│   ├── general/                 # Miscellaneous Ansible utilities
│   ├── k8s/                     # Kubernetes cluster setup using Kubespray
│   └── argocd/                  # ArgoCD playbooks for app lifecycle
```

---

## Infrastructure Provisioning (Terraform)

> **Location**: [`provision.ipynb`](./Vamsi/provision.ipynb)

We use Terraform from a Jupyter notebook on Chameleon to provision:
- 1 controller node (with floating IP)
- 2 worker nodes (internal IPs)
- All nodes are in `private_net_project17` and `private_subnet_project17`

Make sure to:
1. Upload your application credentials to OpenStack.
2. Adjust `terraform.tfvars` to match your project name and keypair.

Run the notebook cell-by-cell to provision your infrastructure.

---

## Kubernetes Cluster Setup (Kubespray + Ansible)

> **Location**: [`ansible/k8s/`](./Vamsi/ansible/k8s/)

We use Kubespray (locked to commit `184b15f`) to bootstrap a production-ready Kubernetes cluster.

### Steps:
1. Fill `inventory.yml` with your node IPs.
2. Activate your Ansible virtual environment:
   ```bash
   source ~/ansible-venv/bin/activate
   ```
3. Run the cluster setup:
   ```bash
   cd Vamsi/ansible/k8s/kubespray
   ansible-playbook -i ../../hosts.yaml --become --become-user=root cluster.yml
   ```

---

## Application Deployment with ArgoCD

> **Playbooks**: [`ansible/argocd/*.yml`](./Vamsi/ansible/argocd/)

The deployment leverages ArgoCD for GitOps-based continuous delivery. Each environment (init, staging, prod, canary) is defined with a separate playbook:

- `argocd_add_platform.yml`: bootstraps ArgoCD
- `workflow_templates_apply.yml`: applies workflow templates
- `argocd_add_staging.yml`, `argocd_add_prod.yml`, `argocd_add_canary.yml`: promote builds to environments

Run them like:
```bash
ansible-playbook -i inventory.yml argocd_add_platform.yml
```

---

## Application Code and Deployment

> **Manifests**:
> - [`dr-dialog-app.yaml`](./Vamsi/dr-dialog-app.yaml)
> - [`service.yaml`](./Vamsi/service.yaml)

These YAML files define the chatbot service, including deployment and NodePort/ClusterIP services. You can apply them manually via `kubectl` or ArgoCD will manage them automatically.

```bash
kubectl apply -f dr-dialog-app.yaml
kubectl apply -f service.yaml
```

---

## Full Deployment Script

> **Location**: [`deploy.sh`](./Vamsi/deploy.sh)

Automates:
- ArgoCD app sync
- Service verification
- Health checks

Run with:
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## How to Run the Project on Chameleon Cloud

1. Log into your JupyterHub space at [jupyter.chameleoncloud.org](https://jupyter.chameleoncloud.org)
2. Clone this repo:
   ```bash
   git clone https://github.com/your-username/Medical-Chatbot-MLOps.git
   cd Medical-Chatbot-MLOps/Vamsi
   ```
3. Run `provision.ipynb` to spin up nodes.
4. SSH into controller node, activate Ansible venv:
   ```bash
   source ~/ansible-venv/bin/activate
   ```
5. Deploy Kubernetes with Ansible + Kubespray.
6. Run ArgoCD playbooks to sync application.
7. Confirm chatbot is live using:
   ```bash
   kubectl get pods
   kubectl get svc
   ```

---

## Final Notes

- Ensure all nodes are reachable via their private IPs inside the cluster.
- If ArgoCD UI is exposed, you can monitor deployments from the browser.
- All YAMLs and playbooks are idempotent and can be re-run.

