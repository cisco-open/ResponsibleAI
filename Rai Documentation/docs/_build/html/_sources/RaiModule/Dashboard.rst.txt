.. _Dashboard:

=============
**Dashboard**
=============


The goal is to provide a generic framework to perform various types of analysis, from data quality assessment to model selection based on performance, fairness, and robustness criteria.


**RAI Dashboard features**
==========================


- Rai dashboards display metrics and critical data and give the user a visual representation of them.
- The Rai dashboard computes all metrics and visualizes the current running code in real time.
- Rai dashboard can handle a wide variety of models and store information for each model, from text-based models to image-based models.
- Rai dashboard automatically determines what metrics are relevant by examining the type of model, data, and task provided by the user.
- With RAI, users can view data at a high level by scanning dashboards like input and output visualizations.


**How To Run RAI Dashboard**
============================

- when the python file ``main.py`` is triggered, Dash is running on server i.e the IP of 127.0.0.1 and the port is 8050, if we follow and click on the URL or copy the URL and paste in the browser, you can access the dashboard.


**Description**: Run the main.py by


.. code-block:: bash

  Dashboard> python .\main.py 

 -Click the following link to access Dashboard.
 -Dash is running on ...




**Example:** 


.. figure:: /images/dash_example.png
   :class: with-border
   :scale: 30 %
   :align: center



**Interaction of RAI Dashboard**
================================

**How it links to RAI project (via redis..)**


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




