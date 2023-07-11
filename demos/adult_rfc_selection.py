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
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.dataset import NumpyData, Dataset
from RAI.db.service import RaiDB
from RAI.utils import df_to_RAI

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
load_dotenv(f'{currentdir}/../.env')

use_dashboard = True
data_path = "../data/adult/"

# loading train and test data
train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)

# convert aggregated data into RAI format
meta, X, y, output = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar", max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

# Create a model to make predictions
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
model = Model(agent=reg, output_features=output, name="cisco_income_ai", predict_fun=reg.predict, predict_prob_fun=reg.predict_proba,
              description="Income Prediction Model", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1}, "time_complexity": "polynomial"}

# setup the dataset
dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})


# initialize RAI 
ai = AISystem("Adult_rfc_selection", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiDB(ai)
    r.reset_data()



#  function to evaluate each model  
def test_model(mdl, name):
    mdl.fit(xTrain, yTrain)
    ai.model.agent = mdl
    ai.model.predict_fun = mdl.predict
    ai.model.predict_prob_fun = mdl.predict_proba
    ai.compute({"test": {"predict": mdl.predict(xTest)}}, tag=name)

    if use_dashboard:
        r.export_metadata()
        r.add_measurement()


test_model(reg, "rfc_entropy")
reg2 = RandomForestClassifier(n_estimators=11, criterion='gini', random_state=0, min_samples_leaf=10, max_depth=3)
test_model(reg2, "rfc_gini")

if use_dashboard:
    r.export_visualizations("test", "test")
