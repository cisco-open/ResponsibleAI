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
# This demo uses the Adults dataset (https://archive.ics.uci.edu/ml/datasets/adult) to show
# how RAI can be used in model selection

# importing modules
import os
import sys
import inspect
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.dataset import NumpyData, Dataset
from RAI.db.service import RaiDB
from RAI.utils import df_to_RAI

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# setup database location
load_dotenv(f'{currentdir}/../.env')

use_dashboard = True
np.random.seed(21)

# Configuration
data_path = f"{currentdir}/../data/adult/"

# loading train and test data
train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)
idx = all_data['race'] != 'White'
all_data['race'][idx] = 'Black'
target_column = "income-per-year"

# convert aggregated data into RAI format
meta, X, y, output = df_to_RAI(all_data, target_column=target_column, normalize="Scalar", max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

# create and train the base model
clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)

# Create a model to make predictions
model = Model(agent=clf, output_features=output, name="cisco_income_ai", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              description="Income Prediction", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

# setup the dataset
dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})

# certificates
cert_location = f"{currentdir}/../RAI/certificates/standard/cert_list.json"

# initialize RAI 
ai = AISystem("AdultDB_two_model", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration, custom_certificate_location=cert_location)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
predictions_train = clf.predict(xTrain)
ai.compute({"test": {"predict": predictions}, "train": {"predict": predictions_train}}, tag="Random Forest 5 Estimator")

if use_dashboard:
    r = RaiDB(ai)
    r.reset_data()
    r.add_measurement()
    r.export_metadata()

mdl = RandomForestClassifier(n_estimators=10, min_samples_leaf=20, max_depth=2)
mdl.fit(xTrain, yTrain)
predictions = mdl.predict(xTest)
predictions_train = mdl.predict(xTrain)
ai.compute({"test": {"predict": predictions}, "train": {"predict": predictions_train}}, tag="Random Forest 10 Estimator")

if use_dashboard:
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")
