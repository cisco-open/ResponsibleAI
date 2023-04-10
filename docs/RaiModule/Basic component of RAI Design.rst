.. _Basic component of RAI Design:

=================================
**Basic component of RAI Design**
=================================

**AISystem**
============

- **Representation**

 - AISystems are the main class users interact with in RAI, they capture key information about an AI. 
 - This information is passed during construction and includes a name, a task type, a MetaDatabase, a Dataset and a Model.


- **Interaction**

 - AISystems make it simple to run computations and get metric values, and are needed to run an Analysis and run Certifications.
 - After making an AISystem, users can use the compute function to generate all relevant metrics related to their model and dataset.
 - The Model, Task Type and MetaDatabase are RAI classes which provide critical information to the AISystem, allowing it to determine which metrics and analyses are relevant. 
 - After computing metrics, users can get retrieve metric values using the get_metric_values function.
 - When provided a network's functions to generate predictions or values, AISystems can use models and run evaluations without requiring user involvement.  


**Example**:

.. container:: toggle, toggle-hidden

    .. admonition:: AI_sys file example

        .. figure:: ../images/aisys.png
           :align: center
           :scale: 40 %



.. important:: **Rai utilization:**
   RAI will utilize AISystem to compute and determine which metrics are relevant (e.g. of the chosen model design).


**Certificates**
================

- **Representation**

 - Certificates allow users to quickly and easily define and communicate standards for AISystems in different domains and tasks.
 - Once a certificate has been added to an AISystem, the AISystem can quickly and easily evaluate whether or not it meets the standards of the certificate allowing for quick yet robust evaluation. 
 - Certificates are written in JSON and can contain logical and relational operators, with the ability to retrieve any metric associated with an AISystem.


- **Interaction**

 - Custom Certificates can be added to an AISystem by passing in a filepath to the certificate file while initializing the AISystem.
 - Certificates will be evaluated when the AISystem calls its compute function. 
 - Certificates can be retrieved by calling the get_certificate_values function on the AISystem.


**Example**:

- **display_name**: Adversarial Bound Bronze Certification.
- **description**: Certifies whether or not the agent is robust against adversarial attacks.


.. container:: toggle, toggle-hidden

    .. admonition:: Certi file example

        .. figure:: ../images/certi.png
           :align: center
           :scale: 30 %




.. important:: **Rai utilization:**
   RAI will carry out detailed analyses (e.g. of the chosen model design) and tests (e.g. robustness, bias, explainability) and define a certification that have Accuracy features.


**Metrics**
===========

- **Representation**

 - Each Metric comes with metadata, including its name and description, as well as a function to compute the metric.
 - Metrics are grouped into MetricGroups, which are collections of Metrics with similar compatibility and functionality.
 - AISystems access metrics through MetricManagers which are responsible for checking compatibility between MetricGroups and AISystems, as well as computing and retrieving specific Metric values.
 - MetricManagers are automatically created and managed by AISystems and are the key to running Metrics and retrieving their values. 

- **Interaction** 

 - Interaction with Metrics are done through MetricManagers. 
 - MetricManagers are capable of quickly finding all MetricGroups compatible with an AISystem. 
 - RAI ensures that dependencies between Metrics are satisfied with no circular dependency issues.  
 - Functionality is provided to search for specific Metrics based on Metric Name, Metric Group Name, Category, and Tags.
 - Metrics are compatible with both whole and batched data. 


**Example**:


.. container:: toggle, toggle-hidden

    .. admonition:: Metric file example

        .. figure:: ../images/metri.png
           :align: center
           :scale: 40 %


.. important:: **Rai utilization:**
   RAI will utilize Metrics to monitor and measures the performance of a model (during training and testing).



**Analysis**
============

- **Representation**

 - While metrics are typically general and simple to calculate, Analyses are finegrained evaluations to run on specific AISystems. 
 - Analyses provide a way for users to quickly and easily run complex experiments compatible with their model, with built in visualizations.
 - Analyses are easy to create allowing users to quickly and easily make their own custom Analyses for their specific needs using any attribute of the AISystem.  

- **Interaction** 
 - Analyses are managed by the AnalysisManger and are given access to the AISystem and Dataset through the RAIRedis class. 
 - Similar to MetricManagers, AnalysisManagers check compatibility between Analyses and AISystems and handle the computation of Analyses.
 - Running specific analyses is done through the run_analysis function. 

**Example**:


.. container:: toggle, toggle-hidden

    .. admonition:: Analysis file example

        .. figure:: ../images/ana.png
           :align: center
           :scale: 40 %


.. important:: **Rai utilization:**
   RAI will carry out detailed analyses and automates report generation and makes data easy to understand.

