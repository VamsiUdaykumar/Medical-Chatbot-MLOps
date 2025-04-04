
## Dr. Dialog: AI for Everyday Health Queries

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->

---
### Value Proposition

#### Status Quo

In today’s healthcare settings, patients often rely on phone calls, emails, or in-person visits to get answers to routine medical questions. This process is slow, inconsistent, and dependent on the availability of clinical staff.

From the **patient's perspective**, this results in:
- Long wait times for basic answers
- Inconvenience, especially outside office hours
- Frustration or anxiety due to lack of timely information

From the **healthcare provider's side**, this creates:
- High volumes of low-complexity queries
- Wasted clinical time on questions that don’t require expertise
- Increased operational costs and reduced focus on critical patient care

---

#### ML System Value

**MediChat** is a cloud-native medical chatbot powered by a **custom-trained large language model (LLM)** that understands and responds to common health queries in natural language. It is designed to be deployed within existing clinical portals or websites and acts as an intelligent first point of contact for patients.

Instead of building a new service, MediChat enhances current healthcare operations by:
- Reducing the manual workload for doctors and nurses
- Providing instant, medically-informed answers to patients
- Operating 24/7 and scaling effortlessly with demand

---

#### Business Metrics

We will evaluate the system based on:
- **Reduction in staff time spent** on repetitive queries (FTE hours saved per week)
- **Patient query resolution rate** via the chatbot (i.e., % of queries fully handled without escalation)
- **Average response time** compared to human-based triage
- **Patient satisfaction score** (e.g., feedback ratings or NPS)

--


### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                     | Responsible for                                         | Link to their commits in this repo |
|--------------------------|---------------------------------------------------------|------------------------------------|
| All team members         | Project idea, value proposition, system design          |                                    |
| Raghu V Hemadri          | Model Training & Infrastructure (Units 4 & 5)           |                                    |                             
| Tejdeep Chippa           | Model Serving & Monitoring (Units 6 & 7)                |                                    |
| Vamsi UK Jonnakuti       | Data Pipeline (Unit 8)                                  |                                    |
| Geetha K Guruju          | Continuous X: CI/CD, Deployment, Infra-as-Code (Unit 3) |                                    |




### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


