
# Dr. Dialog: AI for Everyday Health Queries

## Value Proposition

### Current Status Quo
Traditionally, initial patient assessments and triaging are performed manually by healthcare professionals, such as nurses or general practitioners. This conventional approach often leads to:

- **Extended wait times**: Patients experience delays before receiving initial advice.
- **High operational costs**: Manual triage demands significant human resources.
- **Variable guidance quality**: The consistency and depth of initial evaluations can vary.

### Proposed ML System Advantages
Integrating the ML system into existing healthcare or telemedicine services brings several key improvements:

- **Efficiency and Speed**: Automated processing provides rapid, preliminary assessments based on a patient’s symptoms.
- **Consistency and Personalization**: Leverages a vast dataset and advanced models for consistent, tailored advice.
- **Cost-effectiveness**: Reduces reliance on manual triage, lowering operational costs.
- **Scalability**: Handles high volumes of queries, beneficial in peak or resource-constrained scenarios.

### Business Metrics for Evaluation
Success for this ML system will be measured by:

- **Response Time**: Reduction in the average time to provide a preliminary answer.
- **Accuracy**: Precision in matching symptoms with likely conditions and suggesting appropriate next steps.
- **Patient Satisfaction**: Improvement in satisfaction scores, reflecting relevance and quality of guidance.
- **Operational Efficiency**: Reduction in manual triage workload and staffing costs.

### Integration with Existing Services
The system is designed to plug into existing telemedicine or clinical support tools, enhancing current services without requiring new business models.

---

## Contributors

| Name               | Responsible for                                         | Link to their commits in this repo |
|--------------------|---------------------------------------------------------|------------------------------------|
| All team members   | Project idea, value proposition, system design          |                                    |
| Raghu V Hemadri    | Model Training & Infrastructure (Units 4 & 5)           |                                    |
| Tejdeep Chippa     | Model Serving & Monitoring (Units 6 & 7)                |                                    |
| Vamsi UK Jonnakuti | Data Pipeline (Unit 8)                                  |                                    |
| Geetha K Guruju    | CI/CD, Deployment, Infra-as-Code (Unit 3)              |                                    |

---

## Scale: Data, Model, and Deployment

### Data
Utilizes 21.2 million QA pairs from the [ai-medical-dataset](https://huggingface.co/datasets/ai4bharat/medical-qa), ensuring broad coverage of health topics and symptom descriptions.

### Model
Trains the **LLaMa 3.1 8B** model (8 billion parameters) on a cluster of **4 A100 GPUs**, meeting medium-large scale criteria.

### Deployment
Served via a **Flask-based API** on a single GPU. Optimized for real-time response and future scalability via containerized infrastructure.

---

## Summary of Outside Materials

| Name            | How it was created                                            | Conditions of use                        |
|------------------|---------------------------------------------------------------|------------------------------------------|
| ai-medical-dataset | Curated and published by ruslanmv on Hugging Face (21.2M QA pairs) | Open RAIL-M License                      |
| MedQuAD         | QA pairs from NIH/NLM resources                               | CC BY 4.0                                |
| GPT-2 (baseline) | Pretrained on WebText by OpenAI                              | Open license for research and use        |
| LLaMa 3.1 8B     | Meta AI open release                                          | Research use under Meta's LLaMa license  |

---

## Data Source, Provenance, and Ethical Considerations

- **Dataset Origin**: From ruslanmv, hosted on Hugging Face under the CreativeML Open RAIL-M license.
- **Collection Conditions**: Provided in Parquet format. Precise curation process is unclear; we aim to investigate further.
- **Privacy**: All data is assumed anonymized; we will verify and strip PII if found.
- **Fairness**: We’ll audit for systemic biases and mitigate through preprocessing and evaluation.
- **Ethical Use**: Licensed use is adhered to; disclaimers clarify model limitations and responsibilities.

---

## Infrastructure Requirements

| Requirement       | How many/when                                     | Justification                            |
|-------------------|---------------------------------------------------|------------------------------------------|
| `m1.medium` VMs   | 2 for full project duration                       | Serve API, UI, and log monitoring stack  |
| `gpu_a100`        | 1 node × 6 hrs/week for training                  | Train large-scale LLM (8B params)        |
| Floating IPs      | 1 throughout                                      | Public access for API/UI demos           |
| Block storage     | 20 GB throughout                                  | Datasets, checkpoints, logs              |
| MLflow Server     | 1 container                                       | Experiment and metric tracking           |

---

## Detailed Design Plan

### Model Training and Training Platforms
- **Strategy**: PyTorch Lightning + Ray for training and tuning.
- **Infra**: Trained on 4 × A100 GPUs.
- **Evaluation**: BLEU score, symptom-to-diagnosis match.
- **Extra Credit**: Ray + MLflow integration + distributed tuning.

### Model Serving and Monitoring Platforms
- **Strategy**: Flask API for serving, Prometheus + Grafana for monitoring.
- **Infra**: Single GPU with Docker containers.
- **Metrics**: Latency, error rates, uptime.
- **Extra Credit**: Canary rollout + UI frontend (Streamlit).

### Data Pipeline
- **Strategy**: ETL pipeline from raw QA → cleaned JSONL.
- **Tools**: pandas, spaCy, Python.
- **Extra Credit**: Label auto-tagging + feedback-based re-ingestion.

### Continuous X (CI/CD and IaC)
- **Strategy**: GitHub Actions + Docker + Terraform/bash.
- **Coverage**: Lint, test, deploy steps automated.
- **Extra Credit**: Full pipeline redeployment on model updates.

---
