
# Medical Chatbot MLOps Project: README (my Contributions)

This document summarizes my contributions to the Medical Chatbot MLOps project, in alignment with the course requirements.

---

## Persistent Storage

**Requirement:** Provision persistent storage (block and object) on Chameleon Cloud to store all project-related artifacts.

### My Implementation:

- **Object Storage**:

  - **Tool**: `rclone`, MinIO
  - **Location**: All offline and retraining data are stored in Chameleon’s object store at `chi@tacc`.
  - **Buckets Created**:
    - `/mnt/object/data/dataset-split/` – Contains offline training and validation data.
    - `/mnt/object/data/production/retraining_data_raw/` – Contains raw logs and production data.
    - `/mnt/object/data/production/retraining_data_transformed/` – Contains transformed production data used for retraining.
    - `/mnt/object/artifacts/minio_data/` – Stores all MLflow artifacts, including metrics and model checkpoints.
    - `/mnt/object/artifacts/medical-qa-model/` – Stores only the best model checkpoints selected from training.

  ![Model Artifacts](images/model_artifacts.png)
  ![MinIO MLflow Artifacts](images/minio_artifacts.png)
  ![MinIO File Browser](images/minio_browser.png)

- **Block Storage (KVM @ TACC)**:

  - Provisioned volume and mounted on `/mnt/block`.
  - **Service Using It**: MLflow experiment tracking stores the backend database and artifact metadata here.
  - **Compose Config**:
    - [`docker-compose-persistant-storage.yaml`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/docker/docker-compose-persistant-storage.yaml)
  - **Scripts Provided**:
    - [`block_mount.sh`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/scripts/block_mount.sh)
    - [`object_mount.sh`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/scripts/object_mount.sh)
    - [`kvm_setup.ipynb`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/scripts/kvm_setup.ipynb)

  ![Block Storage Mounted](images/block_storage.png)

- **MLflow UI**: [http://129.114.25.221:8000/](http://129.114.25.221:8000/)
  ![MLflow UI](images/mlflow_ui.png)

---

## Offline Data

**Requirement:** Manage training and validation datasets, transform into appropriate format, and store in object repository.

### My Implementation:

- **Dataset Used**: `lavita/MedQuAD` from HuggingFace

- **ETL Pipeline (Docker-based)**:
  - Extract, Transform, Load using:
    - [`data_preprocessing.py`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/data_preprocessing.py)
    - [`docker-compose-etl.yaml`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/docker/docker-compose-etl.yaml)
    - [`run_etl.sh`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/scripts/run_etl.sh)

- **Tools**: Python, Docker Compose, Bash
- **Environment Setup**:
  - [`requirements.txt`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/requirements.txt)
  - [`Dockerfile`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/Dockerfile)

  ![Object Store Data View](images/object_data.png)
  ![Dataset Split Structure](images/dataset_split.png)

---

## Data Pipeline (Retraining)

**Requirement:** Define pipeline for ingesting, cleaning, and transforming new production data for retraining.

### My Implementation:

- **Directory Structure:**

  ```
  /mnt/object/data/production/
  ├── retraining_data_raw/
  ├── retraining_data_transformed/
  ├── production_data_archive/
  ```

- **Scripts**:
  - [`retraining_data_transform.py`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/retraining_data_transform.py)
  - [`docker-compose-retraining-etl.yaml`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/docker/docker-compose-retraining-etl.yaml)
  - [`run_retraining_etl.sh`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/scripts/run_retraining_etl.sh)

  ![Production Folders](images/production_pipeline.png)

---

## Online Data & Simulation

**Requirement:** Simulate online data streaming using real dataset to mimic real-world inference requests.

### My Implementation:

- **Simulation Dataset**:  
  I initially set aside **20% of the MedQuAD dataset** as unseen data, which was not used during training or validation. This subset was further divided into multiple production sets (`set1`, `set2`, etc.).

- **Simulation Strategy**:
  - Each production set mimics daily usage by users in a real medical QA system.
  - I simulate patient queries arriving sequentially by iterating through the production JSON records.
  - Each request is sent in real time (or with an artificial delay) to the FastAPI inference endpoint.
  - The responses are logged and synced back to the object store for retraining.

- **Script**: [`simulate_online_data.py`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/simulate_online_data.py)

- **Output Example**:
  ```json
  {
    "timestamp": "2025-05-10T13:00:00Z",
    "question": "Can children have asthma symptoms?",
    "model_response": "Yes, children can experience symptoms like coughing and wheezing."
  }
  ```

---

## Interactive Data Dashboard

- **Tool**: Streamlit + Plotly
- **Dashboard UI**: [http://129.114.25.221:8501/](http://129.114.25.221:8501/)
- **Files**:
  - [`dashboard.py`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/dashboard.py)
  - [`requirements.txt`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/requirements.txt)
  - [`Dockerfile`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/Dockerfile)
  - [`docker-compose-dashboard.yaml`](https://github.com/phoenix1881/Medical-Chatbot-MLOps/blob/main/Geetha/docker/docker-compose-dashboard.yaml)

  ![Data Dashboard](images/data_dashboard.png)

