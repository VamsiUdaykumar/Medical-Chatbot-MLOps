
## Title of project

Dr. Dialog: AI for Everyday Health Queries

### Value Proposition

#### Current Status Quo
Traditionally, initial patient assessments and triaging are performed manually by healthcare professionals, such as nurses or general practitioners. This conventional approach often leads to:

- **Extended wait times**: Patients experience delays before receiving initial advice.
- **High operational costs**: Manual triage demands significant human resources.
- **Variable guidance quality**: The consistency and depth of initial evaluations can vary.

#### Proposed ML System Advantages
Integrating the ML system into existing healthcare or telemedicine services brings several key improvements:

- **Efficiency and Speed**: Automated processing provides rapid, preliminary assessments based on a patient’s symptoms.
- **Consistency and Personalization**: Leverages a vast dataset and advanced models for consistent, tailored advice.
- **Cost-effectiveness**: Reduces reliance on manual triage, lowering operational costs.
- **Scalability**: Handles high volumes of queries, beneficial in peak or resource-constrained scenarios.

#### Business Metrics for Evaluation
Success for this ML system will be measured by:

- **Response Time**: Reduction in the average time to provide a preliminary answer.
- **Accuracy**: Precision in matching symptoms with likely conditions and suggesting appropriate next steps.
- **Patient Satisfaction**: Improvement in satisfaction scores, reflecting relevance and quality of guidance.
- **Operational Efficiency**: Reduction in manual triage workload and staffing costs.

#### Integration with Existing Services
The system is designed to plug into existing telemedicine or clinical support tools, enhancing current services without requiring new business models.

### Contributors

| Name                            | Responsible for                                         | Link to their commits in this repo |
|---------------------------------|---------------------------------------------------------|------------------------------------|
| All team members                | Project idea, value proposition, system design          |                                    |
| Raghu V Hemadri                 | Model Training & Infrastructure (Units 4 & 5)           |                                    |
| Tejdeep Chippa                  | Model Serving & Monitoring (Units 6 & 7)                |                                    |
| Vamsi UK Jonnakuti              | Data Pipeline (Unit 8)                                  |                                    |
| Geetha K Guruju                 | CI/CD, Deployment, Infra-as-Code (Unit 3)               |                                    |

### System diagram

The following diagram illustrates the architecture of our ML system, including all core components:

![System Diagram](./architecture-diagram.jpeg)

It includes:
- **Data Pipeline**: ETL from offline and online sources, quality checks, and real-time dashboards.
- **Training Infrastructure**: Distributed model training using 4×A100 GPUs, LLaMa 3.1 8B, MLflow tracking, and fault tolerance.
- **CI/CD Pipeline**: Automates training, evaluation, and deployment using model registry triggers.
- **Model Serving**: Flask-based API deployed on GPU instances with quantized models.
- **Evaluation & Monitoring**: Offline fairness checks, performance dashboards, and feedback loops.

### Summary of outside materials

|              | How it was created                                                  | Conditions of use                        |
|--------------|----------------------------------------------------------------------|------------------------------------------|
| ai-medical-dataset | Curated by ruslanmv, 21.2M QA pairs on Hugging Face            | CreativeML Open RAIL-M License           |
| MedQuAD      | NIH/NLM QA dataset                                                  | CC BY 4.0                                |
| GPT-2        | Pretrained on WebText by OpenAI                                     | Open for research and commercial use     |
| LLaMa 3.1 8B | Meta's open LLM fine-tuned with Ray + DDP on medical dataset        | Research use under Meta license          |

### Summary of infrastructure requirements

| Requirement     | How many/when                         | Justification                                         |
|-----------------|----------------------------------------|-------------------------------------------------------|
| `m1.medium` VMs | 2 for entire project duration          | Host backend API, frontend, monitoring stack          |
| `gpu_a100`      | 1 node × 6 hrs/week for fine-tuning    | Train 8B parameter LLM efficiently                    |
| Floating IPs    | 1 for entire project duration          | Public access to API and frontend                     |
| Block storage   | 20 GB                                  | Store datasets, logs, and model checkpoints           |
| MLflow Server   | 1 container                            | Experiment tracking and metric logging                |

### Detailed design plan

#### Model training and training platforms

- **Unit 4**: Train LLaMa 3.1 8B on 21.2M QA pairs using 4×A100 GPUs with DDP/FSDP.
- **Unit 5**: Hosted on Chameleon with MLFlow tracking and Ray Tune for hyperparameter tuning.
- **Difficulty Points**: Distributed training (Ray Train), checkpointing, fault-tolerant infra, staging-to-production deployment pipeline.

#### Model serving and monitoring platforms

- Flask-based REST API to serve real-time queries
- Targets low-latency with concurrency and async handling
- Optimization via FP16/quantization + graph optimization
- Prometheus and Grafana for logs, throughput, latency, and drift
- Benchmark GPU vs CPU inference and evaluate cost/performance trade-offs

#### Data pipeline

- Offline pipeline for ingesting, cleaning, and storing large-scale QA data
- Online pipeline for capturing real-time user inputs and feeding to retraining queue
- Data versioning and monitoring using dashboards to detect drift and quality issues
- Feedback loop for retraining and continual improvement

#### Continuous X

- CI/CD setup using GitHub Actions + Docker + Terraform
- Canary-based deployment strategy
- Trigger-based automated retraining pipeline
- Offline evaluation covering performance, fairness, bias, failure modes
- Integration with MLFlow for model registry and deployment staging

