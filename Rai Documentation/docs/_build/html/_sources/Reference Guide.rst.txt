.. _Reference Guide:

=============
**Reference**
=============


  
   



**AISystem**
============



    

 - **Class AISystem.**
     
.. py:function:: Class AISystem

AI Systems are the main class users interact with in RAI.
When constructed, AISystems are passed a name, a task type, a MetaDatabase, a Dataset and a Model.
Attributes of the model are used to determine which metrics are relevant.



   
**Parameters.**
     					 -name(str)
                         -task:(str)
                         -meta_database: MetaDatabase,
                         -dataset: Dataset,
                         -model: Model,
                         -enable_certificates: bool = True) -> None:


 Return:None

 Return Type: None

-AIsystem.Initialize

.. code-block:: bash

 Parameters- user_config: dict = {}, custom_certificate_location: str = None
 Return- None
 Return type-None


-get_metric_values

.. code-block:: bash

 Parameters- None
 Return- _last_metric_values
 Return type- dict:



-display_metric_values

.. code-block:: bash

 Parameters- display_detailed
 Default= false
 Return- None
 Return type- None


-certificate_values


 .. code-block:: bash


 Parameters- None
 Return- last_certificate_values
 Return type- dict




-get_data


.. code-block:: bash


 Parameters- data_type:
 Return- dataset
 Return type- list



-get_project_info

.. code-block:: bash

 Parameters- none
 Return- project detials
 Return type-dict


-get_data_summary

.. code-block:: bash

 Parameters- none
 Return- data_summary
 Return type-dict


-Compute

.. code-block:: bash

 Parameters- predictions, tag
 Default -tag=None
 Return- none
 Return type-none


-run_compute

.. code-block:: bash

 Parameters- tag
 Default -tag=None
 Return- none
 Return type-none

-get_metric_info

.. code-block:: bash

 Parameters -None
 Return- none
 Return type-none

-get_certificate_info

.. code-block:: bash

 Parameters -None
 Return- none
 Return type-none


**Certificates**
================

   
Certificate Objects contain information about a particular certificate.
Certificates are automatically loaded in by CertificateManagers and perform evaluation using metric
data they are provided in combination with the certificate data loaded in.
    
- **class Certificate.**


-load_from_json

.. code-block:: bash

 Parameters - json_file
 Return- none
 Return type-none


-Evaluate

.. code-block:: bash

Parameters - metrics, certs
Return- term_values
Return type-list



**Metrics**
===========


Metric class loads in information about a Metric as part of a Metric Group.
Metrics are automatically created by Metric Groups.




- **class Metricmaneger.**



.. code-block:: bash

 Parameters - ai_system ,
 self._time_stamp = None
	        self._sample_count = 0
	        self._last_metric_values = None
	        self._last_certificate_values = None
	        self.ai_system = ai_system
	        self.metric_groups = {}
	        self.user_config = {"fairness": {"priv_group": {}, "protected_attributes": [], "positive_label": 1},
	                            "time_complexity": "exponential"}


 Return- None

 Return type- None



-standardize_user_config

.. code-block:: bash

 Parameters - user_config:
 Return- None
 Return type-None


-Initialize

.. code-block:: bash

 Parameters - user_config: dict = None, metric_groups: list[str] = None, max_complexity: str = "linear"

 Return- None
 Return type- None


-reset_measurements

.. code-block:: bash

 Parameters - None
 Return- None
 Return type-None


-get_metadata

.. code-block:: bash

 Parameters - None
 Return- metric_groups
 Return type-dict:



-get_metric_info_flat

.. code-block:: bash

 Parameters - None
 Return- metric_groups
 Return type-dict:


-compute

.. code-block:: bash

 Parameters - data_dict
 Return- metric_groups
 Return type-dict:

-iterator_compute

.. code-block:: bash

 Parameters - data_dict, preds: dict
 Return- metric_groups
 Return type-dict:



-Search

.. code-block:: bash

 Parameters - query
 Return- metric_groups
 Return type-dict:

- **class Metric.**


-classMetric

.. code-block:: bash

 Parameters - name, config 
 Return- None
 Return type-None


-load_config

.. code-block:: bash

 Parameters - config 
 Return- None
 Return type-None

