# Final Report

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
