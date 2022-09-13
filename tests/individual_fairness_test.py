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


from numpy.testing import assert_almost_equal
from aif360.sklearn.metrics import generalized_entropy_error
import os
import sys
from RAI.dataset import NumpyData, Dataset
from RAI.AISystem import AISystem, Model
from RAI.utils import df_to_RAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
use_dashboard = False
np.random.seed(21)

data_path = "../data/adult/"
train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)
idx = all_data['race'] != 'White'
all_data['race'][idx] = 'Black'

meta, X, y, output = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar", max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
model = Model(agent=clf, output_features=output, name="test_classifier", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              description="Detect Cancer in patients using skin measurements", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})
ai = AISystem("AdultDB_Test1", task='binary_classification', meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)

names = [feature.name for feature in ai.meta_database.features]
df = pd.DataFrame(xTest, columns=names)
df['y'] = yTest

bin_gt_dataset = BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=['race'])

df_preds = pd.DataFrame(xTest, columns=names)
df_preds['y'] = predictions
bin_pred_dataset = BinaryLabelDataset(df=df_preds, label_names=['y'], protected_attribute_names=['race'])

benchmark = ClassificationMetric(bin_gt_dataset, bin_pred_dataset, privileged_groups=[{"race": 1}],
                                 unprivileged_groups=[{"race": 0}])

ai.compute({"test": {"predict": predictions}}, tag="Random Forest")
metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_generalized_entropy_error():
    """Tests that the RAI consistency calculation is correct."""
    gt_series = df['y'].squeeze()
    gt_series.index = df['race']
    assert metrics['individual_fairness']['generalized_entropy_index'] == generalized_entropy_error(gt_series, predictions)


def test_coefficient_of_variation():
    """Tests that the RAI coefficient_of_variation calculation is correct."""
    assert metrics['individual_fairness']['coefficient_of_variation'] == benchmark.coefficient_of_variation()


def test_theil_index():
    """Tests that the RAI theil_index calculation is correct."""
    assert metrics['individual_fairness']['theil_index'] == benchmark.theil_index()
