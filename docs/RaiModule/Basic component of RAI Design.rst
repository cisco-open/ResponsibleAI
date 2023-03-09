.. _Basic component of RAI Design:

=================================
**Basic component of RAI Design**
=================================

**AISystem**
============

- **Representation**

 - It represents the main class, users interact with in RAI.
 - When constructed, AISystems are passed a name, a task type, a MetaDatabase, a Dataset and a Model.


- **Interaction**

 - Single compute accepts predictions and the name of a dataset, and then calculates metrics for that dataset.
 - Compute will tell RAI to compute metric values across each dataset which predictions were made on.
 - Run Compute automatically generates outputs from the model, and compute metrics based on those outputs.


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

 - It represents a class automatically created by AISystems.
 - This class loads a file containing information on which certificates to use, before creating associated Certificate Objects, as well as prompting their associated evaluations.


- **Interaction**

 - Loads all certificates found in the stock certificate file..
 - Loads all certificates found in a custom filepath.


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


**Metric**
==========

- **Representation**

 - It represents to create and Manage various MetricGroups which are compatible with the AISystem. 
 - It is created by the AISystem, and will load in all available MetricGroups compatible with the AISystem. 
 - It also provides functions to run computes for all metric groups, get metadata about metric groups, and get metric values.

- **Interaction** 

 - Find all compatible metric groups.
 - Remove metrics with missing dependencies.
 - Check for circular dependencies.
 - batched_compute check.
 - If data instance of IteratorData, iterate through batches,
 - Searches all metrics. Queries based on Metric Name, Metric Group Name, Category, and Tags.


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

 - It is a method of data analysis that automates analytical model building
 - It analyzes data using machine learning algorithms to predict future outcomes and reveal trends and patterns.

- **Interaction** 

**Example**:


.. container:: toggle, toggle-hidden

    .. admonition:: Analysis file example

        .. figure:: ../images/ana.png
           :align: center
           :scale: 40 %


.. important:: **Rai utilization:**
   RAI will carry out detailed analyses and automates report generation and makes data easy to understand.

