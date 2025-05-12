
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

- **Block Storage (KVM @ TACC)**:

  - Provisioned volume and mounted on `/mnt/block`.

  - **Service Using It**: MLflow experiment tracking stores the backend database and artifact metadata here.

  - **Scripts Provided**:
    - `scripts/block_mount.sh` – Mounts the block volume.
    - `scripts/object_mount.sh` – Mounts the object store.
    - `scripts/kvm_setup.ipynb` – Verifies persistent volume setup and integration with services.

- **MLflow UI**: [http://129.114.25.221:8000/](http://129.114.25.221:8000/)

---

## Offline Data

**Requirement:** Manage training and validation datasets, transform into appropriate format, and store in object repository.

### My Implementation:

- **Dataset Used**: `lavita/MedQuAD` from HuggingFace

- **ETL Pipeline (Docker-based)**:

  - **Extract**:
    - Download dataset via `datasets.load_dataset("lavita/MedQuAD")`
    - Saved in Arrow format in `/data/raw-dataset`
  - **Transform**:
    - Filtered columns: `synonyms`, `question_type`, `question`, `question_focus`, `answer`
    - Removed rows with missing/empty `question` or `answer`
    - Split into `training`, `validation`, and `production` (set1, set2) via deterministic random sampling
  - **Load**:
    - Output stored into `/mnt/object/data/dataset-split/`
  - **Tools**: Python, Docker Compose, Bash
  - **Files Provided**:
    - `data_preprocessing.py` – Applies transformation and splitting logic.
    - `docker/docker-compose-etl.yaml` – Containerized ETL setup.
    - `scripts/run_etl.sh` – Automates launching the ETL container.
    - `requirements.txt`, `Dockerfile` – Environment setup for reproducible ETL pipeline.

- **Data Lineage & Sample**:
  ```json
  {
    "question": "What are the symptoms of asthma?",
    "answer": "Asthma symptoms include shortness of breath, wheezing, and chest tightness."
  }
  ```

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

- **Pipeline Flow**:
  - Extracts new `.json` files from raw directory
  - Cleans and transforms valid entries
  - Saves versioned output and archives raw files

- **Scripts Provided**:
  - `retraining_data_transform.py`
  - `docker-compose-retraining-etl.yaml`
  - `run_retraining_etl.sh`

---

## Online Data & Simulation

**Requirement:** Simulate online data streaming using real dataset to mimic real-world inference requests.

### My Implementation:

- **Script**: `simulate_online_data.py`
- Sends QA requests every few seconds and logs model responses
- Mimics realistic request timing and content

---

## Interactive Data Dashboard

**Requirement:** Build a dashboard to visualize and gain insights into the offline and production data.

### My Implementation:

- **Tool**: Streamlit + Plotly
- **Dashboard UI**: [http://129.114.25.221:8501/](http://129.114.25.221:8501/)
- **Script**: `dashboard.py`
- **Files**:
  - `docker/docker-compose-dashboard.yaml`
  - `requirements.txt`, `Dockerfile`

---

## GitHub Repository Structure (geetha Folder)

```
geetha/
├── dashboard.py
├── data_preprocessing.py
├── retraining_data_transform.py
├── simulate_online_data.py
├── version_tracker.txt
├── requirements.txt
├── Dockerfile
│
├── docker/
│   ├── docker-compose-etl.yaml
│   ├── docker-compose-retraining-etl.yaml
│   ├── docker-compose-dashboard.yaml
│   ├── docker-compose-persistant-storage.yaml
│
├── scripts/
│   ├── block_mount.sh
│   ├── object_mount.sh
│   ├── run_etl.sh
│   ├── run_retraining_etl.sh
│   └── kvm_setup.ipynb
```
