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


from RAI.dataset import Feature, NumpyData, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sklearn
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
output = Feature("Value Prediction", "numeric", "Predicted Value")

meta = MetaDatabase(features)
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


def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (xTest == ai.dataset.data_dict["test"].X).all()
    assert (yTest == ai.dataset.data_dict["test"].y).all()
    assert (xTrain == ai.dataset.data_dict["train"].X).all()
    assert (yTrain == ai.dataset.data_dict["train"].y).all()


def test_explained_variance():
    """Tests that the RAI explained variance calculation is correct."""
    assert metrics['performance_reg']['explained_variance'] == sklearn.metrics.explained_variance_score(yTest, predictions)


def test_mean_absolute_error():
    """Tests that the RAI mean absolute error calculation is correct."""
    assert metrics['performance_reg']['mean_absolute_error'] == sklearn.metrics.mean_absolute_error(yTest, predictions)


def test_mean_absolute_percentage_error():
    """Tests that the RAI mean absolute error calculation is correct."""
    assert metrics['performance_reg']['mean_absolute_percentage_error'] == sklearn.metrics.mean_absolute_percentage_error(yTest, predictions)


def test_mean_gamma_deviance():
    """Tests that the RAI mean gamma deviance calculation is correct."""
    assert metrics['performance_reg']['mean_gamma_deviance'] == sklearn.metrics.mean_gamma_deviance(yTest, predictions)


def test_mean_poisson_deviance():
    """Tests that the RAI mean poisson deviance calculation is correct."""
    assert metrics['performance_reg']['mean_poisson_deviance'] == sklearn.metrics.mean_poisson_deviance(yTest, predictions)


def test_mean_squared_error():
    """Tests that the RAI mean squared error calculation is correct."""
    assert metrics['performance_reg']['mean_squared_error'] == sklearn.metrics.mean_squared_error(yTest, predictions)


def test_mean_squared_log_error():
    """Tests that the RAI mean squared log error calculation is correct."""
    assert metrics['performance_reg']['mean_squared_log_error'] == sklearn.metrics.mean_squared_log_error(yTest, predictions)


def test_median_absolute_error():
    """Tests that the RAI mean squared log error calculation is correct."""
    assert metrics['performance_reg']['median_absolute_error'] == sklearn.metrics.median_absolute_error(yTest, predictions)


def test_r2():
    """Tests that the RAI r2 calculation is correct."""
    assert metrics['performance_reg']['r2'] == sklearn.metrics.r2_score(yTest, predictions)
