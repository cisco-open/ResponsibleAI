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


import os
import sys
import inspect
from RAI.dataset import NumpyData, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
use_dashboard = True

np.random.seed(21)
result = fetch_california_housing(as_frame=True)
target = result.target
df = result.data
df[target.name] = target

meta, X, y, output = df_to_RAI(df, target_column="MedHouseVal")
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1)

reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, output_features=output, name="cisco_income_ai", predict_fun=reg.predict,
              description="Income Prediction AI", model_class="Random Forest Regressor", )
configuration = {"time_complexity": "polynomial"}

dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})
ai = AISystem(name="tabular_regression",  task='regression', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)
ai.compute({"test": {"predict": predictions}}, tag="regression")

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")

ai.display_metric_values("test")
