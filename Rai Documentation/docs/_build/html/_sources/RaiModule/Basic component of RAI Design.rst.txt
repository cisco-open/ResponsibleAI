.. _Basic component of RAI Design:

==================================
**Basic component of RAI Design:**
==================================

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

.. code-block:: bash

 def initialize(self, user_config: dict = {}, custom_certificate_location: str = None, **kw_args):
        self.user_config = user_config
        masks = {"scalar": self.meta_database.scalar_mask, "categorical": self.meta_database.categorical_mask,
                 "image": self.meta_database.image_mask, "text": self.meta_database.text_mask}
        self.dataset.separate_data(masks)
        self.meta_database.initialize_requirements(self.dataset, "fairness" in user_config)
        self.metric_manager = MetricManager(self)
        self.certificate_manager = CertificateManager()
        self.certificate_manager.load_stock_certificates()
        if custom_certificate_location is not None:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)


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


.. code-block:: bash

 {
    "meta":{
        "display_name": "Adversarial Bound Bronze Certification",
        "description": "Certifies whether or not the agent is robust against adversarial attacks.",
        "tags": ["robustness"],
        "level":["1"]
    },

    "condition": {
            "op":"or" ,
            "terms": [              
                [ "&adversarial_validation_tree > adversarial-tree-verification-bound" , ">" , 0.1 ]
            ]
        }
 }     


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

- Find all compatible metric groups


.. code-block:: bash


 for metric_group_name in registry:
            if metric_groups is not None and metric_group_name not in metric_groups:
                continue
            metric_class = registry[metric_group_name]
            self._validate_config(metric_class.config)
            if metric_class.is_compatible(
                    self.ai_system) and metric_group_name in whitelist and metric_group_name not in blacklist:
                compatible_metrics.append(metric_class)
                dependencies[metric_class.config["name"]] = metric_class.config["dependency_list"]
                for dependency in metric_class.config["dependency_list"]:
                    if dependent.get(dependency) is None:
                        dependent[dependency] = []
                    dependent[dependency].append(metric_class.config["name"])


.. important:: **Rai utilization:**
   RAI will utilize Metrics to monitor and measures the performance of a model (during training and testing).



**Analysis**
============

- **Representation**

 - It is a method of data analysis that automates analytical model building
 - It analyzes data using machine learning algorithms to predict future outcomes and reveal trends and patterns.

- **Interaction** 

**Example**:

.. code-block:: bash

 def progress_percent(self, percentage_complete):
        percentage_complete = int(percentage_complete)
        if self.conncetion is not None:
            self.connection(str(percentage_complete))



.. important:: **Rai utilization:**
   RAI will carry out detailed analyses and automates report generation and makes data easy to understand.

