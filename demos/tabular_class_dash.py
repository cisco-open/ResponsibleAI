# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Description
# This demo shows how RAI can be used without the dashboard to calculate and
# report on the metrics for a machine learning task


# importing modules
import os
import sys
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from dotenv import load_dotenv

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.dataset import NumpyData, Dataset
from RAI.db.service import RaiDB
from RAI.utils import df_to_RAI

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

load_dotenv(f'{currentdir}/../.env')

use_dashboard = True
np.random.seed(50)


# Get Dataset
data_path = "../data/adult/"
train_data = pd.read_csv(data_path + "train.csv", header=0, skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0, skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)


# Get X and y data, as well as RAI Meta information from the Dataframe
rai_meta_information, X, y, rai_output_feature = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar")


# Create Data Splits and pass them to RAI
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})


# Create Model and RAIs representation of it
clf = RandomForestClassifier(n_estimators=4, max_depth=2)
model = Model(agent=clf, output_features=rai_output_feature, name="cisco_income_ai", predict_fun=clf.predict,
              predict_prob_fun=clf.predict_proba, description="Income Prediction AI", model_class="RFC")


# Create RAI AISystem to pass all relevant data to RAI
ai = AISystem(name="income_classification",  task='binary_classification', meta_database=rai_meta_information,
              dataset=dataset, model=model)
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}
ai.initialize(user_config=configuration)


# Train the model, generate predictions
clf.fit(xTrain, yTrain)
test_predictions = clf.predict(xTest)
train_predictions = clf.predict(xTrain)


# Pass predictions to RAI
ai.compute({"test": {"predict": test_predictions}, "train": {"predict": train_predictions}}, tag='income_preds')
ai.display_metric_values()

# Connect to the Dashboard
r = RaiDB(ai)
r.reset_data()
r.add_measurement()
r.export_metadata()
r.export_visualizations("test", "test")
