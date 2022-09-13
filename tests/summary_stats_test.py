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


import pandas as pd
import scipy.stats
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

reg = RandomForestRegressor(n_estimators=15, max_depth=20)

output = Feature("Predicted Value", "numeric", "Predicted Value")
model = Model(agent=reg, output_features=output, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

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


def test_num_nan_rows():
    """Tests that the RAI num nan rows calculation is correct."""
    xTestDf = pd.DataFrame(xTest, columns=features)
    assert metrics['summary_stats']['num_nan_rows'] == xTestDf.shape[0] - xTestDf.dropna().shape[0]


def test_percent_Nan_rows():
    """Tests that the RAI percent nan rows calculation is correct."""
    xTestDf = pd.DataFrame(xTest, columns=features)
    assert metrics['summary_stats']['percent_nan_rows'] == (xTestDf.shape[0] - xTestDf.copy().dropna().shape[0])/len(xTestDf)


def test_kstat_1():
    """Tests that the RAI kstat_1 calculation is correct."""
    for i, feature in enumerate(features):
        res = None
        if not feature.categorical:
            res = scipy.stats.kstat(xTest[:, i], 1)
        assert metrics['summary_stats']['kstat_1'][features[i].name] == res


def test_kstat_2():
    """Tests that the RAI kstat_2 calculation is correct."""
    for i, feature in enumerate(features):
        res = None
        if not feature.categorical:
            res = scipy.stats.kstat(xTest[:, i], 2)
        assert metrics['summary_stats']['kstat_2'][features[i].name] == res


def test_kstat_3():
    """Tests that the RAI kstat_3 calculation is correct."""
    for i, feature in enumerate(features):
        res = None
        if not feature.categorical:
            res = scipy.stats.kstat(xTest[:, i], 3)
        assert metrics['summary_stats']['kstat_3'][features[i].name] == res


def test_kstat_4():
    """Tests that the RAI kstat_1 calculation is correct."""
    for i, feature in enumerate(features):
        res = None
        if not feature.categorical:
            res = scipy.stats.kstat(xTest[:, i], 4)
        assert metrics['summary_stats']['kstat_4'][features[i].name] == res


def test_kstatvar():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        res = None
        if not feature.categorical:
            res = scipy.stats.kstatvar(xTest[:, i])
        assert metrics['summary_stats']['kstatvar'][features[i].name] == res


def test_iqr():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        res = None
        if not feature.categorical:
            res = scipy.stats.iqr(xTest[:, i])
        assert metrics['summary_stats']['iqr'][features[i].name] == res


def test_bayes_mvs():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        res = {"mean": [None], "var": [None], "std": [None]}
        if not feature.categorical:
            res["mean"], res["var"], res["std"] = scipy.stats.bayes_mvs(xTest[:, i])
        assert metrics['summary_stats']['bayes_mean_avg'][features[i].name] == res['mean'][0]
        assert metrics['summary_stats']['bayes_variance_avg'][features[i].name] == res['var'][0]
        assert metrics['summary_stats']['bayes_std_avg'][features[i].name] == res['std'][0]


def test_frozen_mvs():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        res_mean = {"mean": None, "var": None, "std": None}
        res_var = {"mean": None, "var": None, "std": None}
        res_std = {"mean": None, "var": None, "std": None}
        if not feature.categorical:
            mean, var, std = scipy.stats.mvsdist(xTest[:, i])
            res_mean['mean'] = mean.mean()
            res_mean['var'] = mean.var()
            res_mean['std'] = mean.std()
            res_var['mean'] = var.mean()
            res_var['var'] = var.var()
            res_var['std'] = var.std()
            res_std['mean'] = std.mean()
            res_std['var'] = std.var()
            res_std['std'] = std.std()
        assert metrics['summary_stats']['frozen_mean_mean'][features[i].name] == res_mean['mean']
        assert metrics['summary_stats']['frozen_mean_variance'][features[i].name] == res_mean['var']
        assert metrics['summary_stats']['frozen_mean_std'][features[i].name] == res_mean['std']
        assert metrics['summary_stats']['frozen_variance_mean'][features[i].name] == res_var['mean']
        assert metrics['summary_stats']['frozen_variance_variance'][features[i].name] == res_var['var']
        assert metrics['summary_stats']['frozen_variance_std'][features[i].name] == res_var['std']
        assert metrics['summary_stats']['frozen_std_mean'][features[i].name] == res_std['mean']
        assert metrics['summary_stats']['frozen_std_variance'][features[i].name] == res_std['var']
        assert metrics['summary_stats']['frozen_std_std'][features[i].name] == res_std['std']
