.. _Getting Started:


===================
**Getting Started**
===================

Here is a quick example of how RAI can be used without the dashboard to calculate and report on the metrics for a machine learning task

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

**Get Dataset**

.. code-block:: python


 data_path = "../data/adult/"
 train_data = pd.read_csv(data_path + "train.csv", header=0, skipinitialspace=True, na_values="?")
 test_data = pd.read_csv(data_path + "test.csv", header=0, skipinitialspace=True, na_values="?")
 all_data = pd.concat([train_data, test_data], ignore_index=True)




.. code-block:: python



        age         workclass  fnlwgt  education  ...  capital-loss hours-per-week native-country income-per-year
 0       39         State-gov   77516  Bachelors  ...             0             40  United-States           <=50K
 1       50  Self-emp-not-inc   83311  Bachelors  ...             0             13  United-States           <=50K
 2       38           Private  215646    HS-grad  ...             0             40  United-States           <=50K
 3       53           Private  234721       11th  ...             0             40  United-States           <=50K
 4       28           Private  338409  Bachelors  ...             0             40           Cuba           <=50K
 ...    ...               ...     ...        ...  ...           ...            ...            ...             ...
 48837   39           Private  215419  Bachelors  ...             0             36  United-States           <=50K
 48838   64               NaN  321403    HS-grad  ...             0             40  United-States           <=50K



**Get X and y data, as well as RAI Meta information from the Dataframe**


.. code-block:: bash

 rai_meta_information, X, y, rai_output_feature = df_to_RAI(all_data, 
 target_column="income-per-year", normalize="Scalar")
   

**Create Data Splits and pass them to RAI**

.. code-block:: bash


 xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
 dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})


.. code-block:: python


   [[-0.49539098  4.         -0.65011375 ... -0.21878026  1.58752287
  38.        ]
 [ 0.33682491  2.         -0.80034375 ... -0.21878026 -0.07812006
  13.        ]
 [-1.25195088  5.         -1.56515356 ... -0.21878026 -1.74376299
  38.        ]
 ...
 [-1.55457484  2.          0.76496379 ... -0.21878026 -0.91094152
  38.        ]



**Create Model and RAIs representation of it**

.. code-block:: bash

 clf = RandomForestClassifier(n_estimators=4, max_depth=6)
 model = Model(agent=clf, output_features=rai_output_feature, 
 name="cisco_income_ai", predict_fun=clf.predict,
 predict_prob_fun=clf.predict_proba, 
 description="Income Prediction AI", model_class="RFC")



**Create RAI AISystem to pass all relevant data to RAI**

.. code-block:: bash

 ai = AISystem(name="income_classification",  task='binary_classification', 
              meta_database=rai_meta_information,
              dataset=dataset, model=model)

 configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                 "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}
 ai.initialize(user_config=configuration)



.. code-block:: python

 RAI.AISystem.ai_system.AISystem object at 0x00000228E34222E0

 {'fairness': {'priv_group': {'race': {'privileged': 1, 'unprivileged': 0}}, 'protected_attributes': ['race'], 'positive_label': 1}, 'time_complexity': 'polynomial'}
 


**Train the model, generate predictions**

.. code-block:: bash

 clf.fit(xTrain, yTrain)
 test_predictions = clf.predict(xTest)


.. code-block:: python

 
 Analysis created
 ==== Group Fairness Analysis Results ====
 1 of 4 tests passed.

 Statistical Parity Difference Test:
 This metric is The difference of the rate of favorable outcomes received by the unprivileged group to the privileged group.    
 The idea value is 0.0.
 It's value of -0.11160752641979553 is not between between 0.1 and -0.1 indicating that there is unfairness.

 Equal Opportunity Difference Test:
 This metric is The difference of true positive rates between the unprivileged and the privileged groups.
 The true positive rate is the ratio of true positives to the total number of actual positives for a given group.
 The ideal value is 0. A value of < 0 implies higher benefit for the privileged group and a value > 0 implies higher benefit for the unprivileged group.
 It's value of -0.12121212121212122 is not between between 0.1 and -0.1 indicating that there is unfairness.