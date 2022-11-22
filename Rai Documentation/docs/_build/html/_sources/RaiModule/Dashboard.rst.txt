.. _Dashboard:

=============
**Dashboard**
=============


The goal is to provides a generic framework to perform various types of analysis from data quality assessment to model selection based on performance, fairness and robustness criteria.

**RAI Dashboard features**
==========================

- A Rai dashboard is a data visualization tool that tracks, analyzes, and display metrics, and critical data.
- Rai dashboard handles all metric computation and perform live visualizations of currently running code on its dashboard. 
- Rai dashboard can handle variety of model and store information of each model from text based models to image based models.
- Rai dashboard automatically determine what metrics are relevant by examining the type of model, data and task provided by the user.
- Rai allows users to quickly scan from dashboard like input and output visualization, obtaining a high-level view of data.




**How To Run RAI Dashboard**
============================

- when the python file ``main.py`` is triggered, Dash is running on server i.e the IP of 127.0.0.1 and the port is 8050, if we follow and click on the URL or copy the URL and paste in the browser, you can access the dashboard.


**Description**: Run the main.py by


.. code-block:: bash

  Dashboard> python .\main.py 

 -Click the following link to access Dashboard.
 -Dash is running on ...



**Example**:

.. code-block:: bash

  >>>INFO:redis_util:channel subscribed
     INFO:redis_util:channel subscribed ai_requests
     INFO:redis_util:changing current project from None to AdultDB_two_model
     current project name:  AdultDB_two_model
     Current proj name:  AdultDB_two_model
     Dash is running on http://127.0.0.1:8050/


**Interaction of RAI Dashboard**
================================

**How it links to RAI project (via redis..)**

- host ="localhost", port=6379, db=0
- We connect redis at localhost i.e 127.0.0.1 on default port 6379 at db 0





.. figure:: /images/Dashboard_page.png
   :class: with-border
   :alt: RAI Dashboard Homepage
   :scale: 40 %
   :align: center

   RAI Dashboard Homepage.






**How it shows the Metrics, Certificates and Analysis page**



- **Metrics:**


.. figure:: /images/Metricpage.png
   :class: with-border
   :alt: Metric Details Page
   :scale: 40 %
   :align: center

   Metric Details Page.

   



- **Certificates:** 



.. figure:: /images/certificate.png
   :class: with-border
   :alt: Certificate Page
   :scale: 40 %
   :align: center

   Certificate Page.


- **Analysis:**


.. figure:: /images/Analysispage.png
   :class: with-border
   :alt: Analysis Page
   :scale: 40 %
   :align: center

   Analysis Page.





**How to contribute and extend RAI**
====================================
