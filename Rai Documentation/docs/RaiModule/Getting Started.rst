.. _Getting Started:


===================
**Getting Started**
===================

Here is a quick example of how RAI can be used without the dashboard to calculate and report on the metrics for a machine learning task.

.. code-block:: bash

 import os
 import sys
 import inspect
 import pandas as pd
 from sklearn.model_selection import train_test_split
 from RAI.AISystem import AISystem, Model
 from RAI.Analysis import AnalysisManager
 from RAI.dataset import NumpyData, Dataset
 from RAI.utils import df_to_RAI
 import numpy as np
 from sklearn.ensemble import RandomForestClassifier

- It starts by importing the necessary libraries



.. figure:: ../images/getting_started_demo.gif
   :align: center
   :scale: 90 %

   Getting_started_demo 



