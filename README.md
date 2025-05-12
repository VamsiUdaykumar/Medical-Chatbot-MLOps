
## Dr. Dialog: AI for Everyday Health Queries

### Value Proposition

#### Current Status Quo

Patients accessing their health data through MyChart or the Mayo Clinic Mobile App often encounter complex medical terminology in their lab reports, imaging summaries, or physician notes. This can result in:

* **Confusion and Misinterpretation**: Medical jargon is difficult for non-experts to understand.
* **Increased support burden**: Patients frequently contact clinical staff for clarification, diverting resources from urgent care.
* **Reduced patient empowerment**: Limited understanding of their own records hinders informed health decisions.

#### Proposed ML System Advantages

The *Report Explainer and Medical Q\&A Assistant* addresses these pain points by offering:

* **Terminology Clarification**: Automatically explains complex medical terms and report sections in plain language.
* **General Medical Q\&A**: Provides reliable, vetted answers to non-personalized medical queries.
* **User Empowerment**: Supports patient health literacy without replacing professional advice or accessing private medical data.
* **Operational Relief**: Reduces the volume of non-critical queries to healthcare professionals.

#### Business Metrics for Evaluation

Impact of the assistant will be measured using:

* **User Engagement**: Increased interaction rates within MyChart and the mobile app.
* **Support Deflection**: Reduction in routine clarifications routed to clinical staff.
* **Patient Understanding**: Improved health literacy scores through feedback or survey mechanisms.
* **Time to Clarity**: Decrease in time taken by users to understand medical reports.

#### Integration with Mayo Clinic Digital Services

This module can be embedded seamlessly into MyChart and the Mayo Clinic App as a non-intrusive, assistive layer. It complements existing functionalities by helping patients better understand their health information, without accessing or processing private patient data. Over time, this assistant can evolve as a modular component in a broader AI-powered digital care ecosystem.


### Contributors

| Name                            | Responsible for                                         | Link to their commits in this repo |
|---------------------------------|---------------------------------------------------------|------------------------------------|
| All team members                | Project idea, value proposition, system design          |                                    |
| Raghu V Hemadri                 | Model Training & Infrastructure (Units 4 & 5)           |                                    |
| Tejdeep Chippa                  | Model Serving & Monitoring (Units 6 & 7)                |                                    |
| Geetha K Guruju              | Data Pipeline (Unit 8)                                  |                                    |
| Vamsi UK Jonnakuti                | CI/CD, Deployment, Infra-as-Code (Unit 3)               |                                    |

### System diagram

The following diagram illustrates the architecture of our ML system, including all core components:

![System Diagram](./architecture-diagram.jpeg)

It includes:
- **Data Pipeline**: ETL from offline and online sources, data quality checks, and storage for training input.
- **Training Infrastructure**: Distributed model training using 4×A100 GPUs for LLaMa 3, with fault tolerance and experiment tracking.
- **CI/CD Pipeline**: Automates training and deployment using model registry and retraining triggers.
- **Model Serving**: Production model served via Flask API on GPU, with optional model optimization.
- **Evaluation & Monitoring**: Load testing, canary testing, offline evaluation (bias/fairness), monitoring dashboards, and user feedback loop.

### Summary of outside materials

|              | How it was created                                                  | Conditions of use                        |
|--------------|----------------------------------------------------------------------|------------------------------------------|
| MedQuAD      | NIH/NLM QA dataset                                                  | CC BY 4.0                                |
| TinyLLaMa 1.1M | Meta's open LLM fine-tuned with Ray + DDP on medical dataset        | Research use under Meta license          |

### Summary of infrastructure requirements

| Requirement     | How many/when                         | Justification                                         |
|-----------------|----------------------------------------|-------------------------------------------------------|
| `m1.medium` VMs | 2 for entire project duration          | Host API, dashboard, and monitoring                   |
| `gpu_a100`      | 1 node × 6 hrs/week for fine-tuning    | Train large-scale LLM efficiently                    |
| Floating IPs    | 1 for entire project duration          | Enable public access to services                      |
| Block storage   | 20 GB                                  | Dataset storage, logs, model artifacts                |
| MLflow Server   | 1 container                            | Experiment and performance tracking                   |

### Detailed design plan

#### Model training and training platforms

- **Unit 4**: Fine-tune TinyLLaMa 1.1B using 2×A100 GPUs on medical Q&A dataset.
- **Unit 5**: Use Ray for distributed training; MLFlow for experiment tracking.
- **Extra Credit**: Fault tolerance, checkpointing, automated experiment management.

#### Model serving and monitoring platforms

- Serve model via Flask-based API on GPU.
- Optimize model using quantization.
- Monitor performance (latency, throughput, drift) with Prometheus + Grafana.
- Evaluate rollout performance using canary testing and feedback loops.

#### Data pipeline

- Offline and online data ingested and cleaned through ETL.
- Store QA pairs and online symptom input in centralized storage.
- Quality dashboard to monitor data integrity.

#### Continuous X

- Automated CI/CD pipeline triggers training and deployment.
- Canary and staging environments for controlled rollout.
- Model registry and feedback-driven retraining pipeline.


# Final Report

## Tejdeep

###  Serving from an API endpoint

-   **Endpoint code**: [`flask_app/app.py`](flask_app/app.py)
  
        
    -   Includes safety filtering (BART-MNLI) → TinyLLaMA inference →  
        post-generation filtering.
        

----------

###  Customer-specific requirements

- Final Metrics we thought would be useful
- Code in Model_optimisations.ipynb
- Model Size: 4400.19 MB 
- Median Latency: 19.37 ms

Our target users are individuals with no medical background who seek to understand complex medical terms in their diagnostic reports or discharge summaries. The chatbot is expected to:

-   **Interpret and explain** medical jargon in layman’s terms.
    
-   **Reject and redirect** any prompts requesting diagnostic conclusions, drug prescriptions, or treatment advice.
    
-   **Ensure fast response time**, especially for queries coming from mobile or low-latency environments.
    
-   **Operate under strict memory and latency constraints**, as real-time user experience is a core requirement.
    
-   **Provide safe and responsible outputs**, always warning users to consult licensed professionals.
    

To meet these requirements, we benchmarked our model (TinyLLaMA variant of Lit-GPT) and observed a median latency of **~19.37 ms** and **throughput of ~51.67 req/sec** for single inference. This ensures rapid turnaround even under moderate to heavy load.
----------

###  Model-level optimizations

-   **Dynamic INT8 quantisation** script:  
    `model_optimisations.ipynb`
    

   

----------

### System-level optimizations

 - **Dynamic Batching** - Use dynamic_app.py
    

----------

### Offline evaluation suite

-   All tests live in `offline_eval.ipynb`
    

----------

### Load test in staging

-  done in `system_optimisations.ipynb`
    
    

----------

### Business-specific evaluation (design only)

For our use case, a conventional NLP accuracy metric like BLEU or ROUGE is less relevant. Instead, we define a business-centric evaluation focused on **three key axes**:

1.  **Comprehensibility Score**: Does the model correctly simplify and clarify the medical jargon for a non-expert? This can be tested using human evaluation or automated readability metrics (e.g., Flesch-Kincaid Grade Level).
    
2.  **Safety Filter Precision**: How well does the model avoid giving prohibited responses (e.g., medical advice, prescriptions)? We define this as the percentage of intercepted and rejected unsafe questions.
    
3.  **Latency SLA Satisfaction Rate**: Percentage of requests served under our SLA of **< 25 ms** for 95th percentile latency. This is a direct measure of real-time viability.
    

----------

### Online monitoring 

-   **Grafana dashboards** used in docker

# Training Details (Raghu)
[View detailed training results/instructions](./Raghu/README.md)
