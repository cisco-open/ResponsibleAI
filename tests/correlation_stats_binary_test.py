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
import scipy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x, y = load_breast_cancer(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)
val_0_count = 20

nums = np.ones((xTrain.shape[0], 1))
nums[:val_0_count] = 0
xTrain = np.hstack((xTrain, nums))
xTrain = np.hstack((xTrain, nums))

nums = np.ones((xTest.shape[0], 1))
nums[:val_0_count] = 0
xTest = np.hstack((xTest, nums))
xTest = np.hstack((xTest, nums))

features_raw = load_breast_cancer().feature_names
features = []

for feature in features_raw:
    features.append(Feature(feature, "numeric", feature))
features.append(Feature("race", "numeric", "race value", categorical=True, values={0: "black", 1: "white"}))
features.append(Feature("gender", "numeric", "race value", categorical=True, values={1: "male", 0: "female"}))

training_data = NumpyData(xTrain, yTrain)
test_data = NumpyData(xTest, yTest)
dataset = Dataset({"train": training_data, "test": test_data})
meta = MetaDatabase(features)

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
output = Feature("Cancer Prediction", "numeric", "Cancer Prediction", categorical=True,
                 values={0: "No Cancer", 1: "Cancer"})
model = Model(agent=rfc, output_features=output, name="cisco_cancer_ai", predict_fun=rfc.predict,
              predict_prob_fun=rfc.predict_proba, model_class="Random Forest Classifier")

configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "positive_label": 1}}
ai = AISystem("cancer_detection", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

rfc.fit(xTrain, yTrain)
predictions = rfc.predict(xTest)

ai.compute({"test": {"predict": predictions}}, tag="binary classification")

metrics = ai.get_metric_values()
metrics = metrics["test"]


def test_point_biserial_r():
    """Tests that the RAI relfreq calculation is correct."""
    for i, feature in enumerate(features):
        res = {}
        if not feature.categorical:
            res = scipy.stats.pointbiserialr(xTest[:, i], yTest)
            res = {"correlation": res.correlation, "pvalue": res.pvalue}
        assert metrics['correlation_stats_binary']['point_biserial_r'][feature.name] == res
