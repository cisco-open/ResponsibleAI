.. image:: https://raw.githubusercontent.com/cisco-open/ResponsibleAI/main/docs/images/rai_logo_blue3.png

RAI is a python library that is designed to help AI developers in
various aspects of responsible AI development. It consists of a core API
and a corresponding web-based dashboard application. RAI can easily be
integrated into AI development projects and measures various metrics for
an AI project during each phase of AI development, from data quality
assessment to model selection based on performance, fairness and
robustness criteria. In addition, it provides interactive tools and
visualizations to understand and explain AI models and provides a
generic framework to perform various types of analysis including
adversarial robustness.

Note: - The dashboard GUI is currently designed around 1920x1080
displays

Documentation
=============

Documentation is available online:
https://responsibleai.readthedocs.io/en/latest/

To Install
==========

1) The project uses sqlite as storage and requires no additional
   installation for this.

    ``pip install rai``

2) If you need to use the dashboard as well you must clone the project
   locally and install the additional requirements

    ``pip install -r requirements.txt``

Demos:
======

::

   We have added a few demo projects to showcase some of the capabilities of RAI.
   to run any of the demos please use 'python demo_filename'. For instance : 
   python ./demos/adult_demo_grid_search.py

   below is a short description of the provided demos:

   File: adult_demo_grid_search.py 
   Description: This demo uses the Adults dataset (https://archive.ics.uci.edu/ml/datasets/adult) to show how RAI can be used in model selection

   File: image_class_analysis.py 
   Description: this demo uses Cifar10 dataset and shows how RAI can be used to evaluate image classification tasks

   File: image_class_training.py 
   Description: this demo uses Cifar10 dataset and shows how RAI can be used monitor image processing tasks during training


   File: tabular_class_console.py 
   Description: this demo shows how RAI can be used without the dashboard to calculate and report on the metrics for a machine learning task

   File: text.py 
   File: text_output.py 
   Description: these demos show how RAI and its dashboard can be used for evaluating the natural language modeling tasks

    

Cisco Research, Emerging Tech and Incubations,

Cisco Systems Inc. 
