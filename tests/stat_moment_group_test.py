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


import scipy.stats
from numpy.testing import assert_almost_equal
from RAI.dataset import Feature, NumpyData, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

use_dashboard = False
np.random.seed(21)

dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})

features = [
    Feature("MedInc", 'numeric', "Median Income"),
    Feature("HouseAge", 'numeric', "Median House age in Block Group"),
    Feature("AveRooms", 'numeric', "Average number of rooms per household"),
    Feature("AveBedrms", 'numeric', "Average number of bedrooms per household"),
    Feature("Population", 'numeric', "Block group population"),
    Feature("AveOccup", 'numeric', "Average Number of Household members"),
    Feature("Latitude", 'numeric', "Block group Latitude"),
    Feature("Longitude", 'numeric', "Block group Longitude")
]
meta = MetaDatabase(features)

output = Feature("Predicted value", "numeric", "Predicted value")

reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, output_features=output, predict_fun=reg.predict, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", task='regression', meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

ai.compute({"test": {"predict": predictions}}, tag="regression")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_moment_1():
    """Tests that the RAI moment 1 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert_almost_equal(scipy.stats.moment(xTest[:, i], 1), metrics['stat_moment_group']['moment_1'][features[i].name], 4)


def test_moment_2():
    """Tests that the RAI moment 2 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert_almost_equal(scipy.stats.moment(xTest[:, i], 2), metrics['stat_moment_group']['moment_2'][features[i].name], 4)


def test_moment_3():
    """Tests that the RAI moment 3 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert_almost_equal(scipy.stats.moment(xTest[:, i], 3), metrics['stat_moment_group']['moment_3'][features[i].name], 4)
