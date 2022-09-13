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


import math
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
from numpy.testing import assert_almost_equal

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

model = Model(agent=clf, output_features=output, name="Cancer detection AI", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              description="Detect Cancer in patients using skin measurements", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}}, "positive_label": 1},
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


def test_average_abs_odds_difference():
    """Tests that the RAI average abs odds difference calculation is correct."""
    assert metrics['prediction_fairness']['average_odds_difference'] == benchmark.average_odds_difference()


def test_between_all_groups_coefficient_of_variation():
    """Tests that the RAI between_all_groups_coefficient_of_variation calculation is correct."""
    assert metrics['prediction_fairness']['between_all_groups_coefficient_of_variation'] == benchmark.between_all_groups_coefficient_of_variation()


def test_between_all_groups_generalized_entropy_index():
    """Tests that the RAI between_all_groups_generalized_entropy_index calculation is correct."""
    assert metrics['prediction_fairness']['between_all_groups_generalized_entropy_index'] == benchmark.between_all_groups_generalized_entropy_index()


def test_between_all_groups_theil_index():
    """Tests that the RAI between_all_groups_theil_index calculation is correct."""
    assert metrics['prediction_fairness']['between_all_groups_theil_index'] == benchmark.between_all_groups_theil_index()


def test_between_group_coefficient_of_variation():
    """Tests that the RAI between_group_coefficient_of_variation calculation is correct."""
    assert metrics['prediction_fairness']['between_group_coefficient_of_variation'] == benchmark.between_group_coefficient_of_variation()


def test_between_group_generalized_entropy_index():
    """Tests that the RAI between_group_generalized_entropy_index calculation is correct."""
    assert metrics['prediction_fairness']['between_group_generalized_entropy_index'] == benchmark.between_group_generalized_entropy_index()


def test_between_group_theil_index():
    """Tests that the RAI between_group_theil_index calculation is correct."""
    assert metrics['prediction_fairness']['between_group_theil_index'] == benchmark.between_group_theil_index()


def test_coefficient_of_variation():
    """Tests that the RAI coefficient_of_variation calculation is correct."""
    assert metrics['prediction_fairness']['coefficient_of_variation'] == benchmark.coefficient_of_variation()


def test_consistency():
    """Tests that the RAI consistency calculation is correct."""
    assert_almost_equal(metrics['prediction_fairness']['consistency'], benchmark.consistency()[0], 1)


def test_differential_fairness_bias_amplification():
    """Tests that the RAI differential_fairness_bias_amplification calculation is correct."""
    assert metrics['prediction_fairness']['differential_fairness_bias_amplification'] == benchmark.differential_fairness_bias_amplification()


def test_error_rate():
    """Tests that the RAI error_rate calculation is correct."""
    assert metrics['prediction_fairness']['error_rate'] == benchmark.error_rate()


def test_error_rate_difference():
    """Tests that the RAI error_rate_difference calculation is correct."""
    assert metrics['prediction_fairness']['error_rate_difference'] == benchmark.error_rate_difference()


def test_error_rate_ratio():
    """Tests that the RAI error_rate_difference calculation is correct."""
    assert metrics['prediction_fairness']['error_rate_ratio'] == benchmark.error_rate_ratio()


def test_false_discovery_rate():
    """Tests that the RAI false_discovery_rate calculation is correct."""
    assert metrics['prediction_fairness']['false_discovery_rate'] == benchmark.false_discovery_rate()


def test_false_discovery_rate_difference():
    """Tests that the RAI false_discovery_rate_difference calculation is correct."""
    assert metrics['prediction_fairness']['false_discovery_rate_difference'] == benchmark.false_discovery_rate_difference()


def test_false_discovery_rate_ratio():
    """Tests that the RAI false_discovery_rate calculation is correct."""
    print("ITS MY TEST AHHH: ", benchmark.false_discovery_rate_ratio())

    assert metrics['prediction_fairness']['false_discovery_rate_ratio'] == benchmark.false_discovery_rate_ratio()


def test_false_discovery_rate_ratio():
    """Tests that the RAI false_discovery_rate calculation is correct."""
    result = benchmark.false_discovery_rate_ratio()
    if math.isnan(result):
        result = None
    assert metrics['prediction_fairness']['false_discovery_rate_ratio'] == result


def test_false_negative_rate():
    """Tests that the RAI false_negative_rate calculation is correct."""
    assert metrics['prediction_fairness']['false_negative_rate'] == benchmark.false_negative_rate()


def test_false_negative_rate_difference():
    """Tests that the RAI false_negative_rate_difference calculation is correct."""
    assert metrics['prediction_fairness']['false_negative_rate_difference'] == benchmark.false_negative_rate_difference()


def test_false_negative_rate_ratio():
    """Tests that the RAI false_negative_rate_ratio calculation is correct."""
    assert metrics['prediction_fairness']['false_negative_rate_ratio'] == benchmark.false_negative_rate_ratio()


def test_false_negative_rate():
    """Tests that the RAI false_negative_rate calculation is correct."""
    assert metrics['prediction_fairness']['false_negative_rate'] == benchmark.false_negative_rate()


def test_generalized_entropy_index():
    """Tests that the RAI generalized_entropy_index calculation is correct."""
    assert metrics['prediction_fairness']['generalized_entropy_index'] == benchmark.generalized_entropy_index()


def test_generalized_true_negative_rate():
    """Tests that the RAI generalized_true_negative_rate calculation is correct."""
    assert metrics['prediction_fairness']['generalized_true_negative_rate'] == benchmark.generalized_true_negative_rate()


def test_generalized_true_positive_rate():
    """Tests that the RAI generalized_true_positive_rate calculation is correct."""
    assert metrics['prediction_fairness']['generalized_true_positive_rate'] == benchmark.generalized_true_positive_rate()


def test_negative_predictive_value():
    """Tests that the RAI negative_predictive_value calculation is correct."""
    assert metrics['prediction_fairness']['negative_predictive_value'] == benchmark.negative_predictive_value()


def test_num_false_negatives():
    """Tests that the RAI num_false_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_false_negatives'] == benchmark.num_false_negatives()


def test_num_false_positives():
    """Tests that the RAI num_false_positives calculation is correct."""
    assert metrics['prediction_fairness']['num_false_positives'] == benchmark.num_false_positives()


def test_num_generalized_false_negatives():
    """Tests that the RAI num_generalized_false_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_generalized_false_negatives'] == benchmark.num_generalized_false_negatives()


def test_num_generalized_false_positives():
    """Tests that the RAI num_generalized_false_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_generalized_false_positives'] == benchmark.num_generalized_false_positives()


def test_num_generalized_true_negatives():
    """Tests that the RAI num_generalized_true_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_generalized_true_negatives'] == benchmark.num_generalized_true_negatives()


def test_num_generalized_true_positives():
    """Tests that the RAI num_generalized_true_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_generalized_true_positives'] == benchmark.num_generalized_true_positives()


def test_num_instances():
    """Tests that the RAI num_instances calculation is correct."""
    assert metrics['prediction_fairness']['num_instances'] == benchmark.num_instances()


def test_num_negatives():
    """Tests that the RAI num_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_negatives'] == benchmark.num_negatives()


def test_num_positives():
    """Tests that the RAI num_positives calculation is correct."""
    assert metrics['prediction_fairness']['num_positives'] == benchmark.num_positives()


def test_num_pred_negatives():
    """Tests that the RAI num_pred_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_pred_negatives'] == benchmark.num_pred_negatives()


def test_num_pred_positives():
    """Tests that the RAI num_pred_positives calculation is correct."""
    assert metrics['prediction_fairness']['num_pred_positives'] == benchmark.num_pred_positives()


def test_num_true_negatives():
    """Tests that the RAI num_true_negatives calculation is correct."""
    assert metrics['prediction_fairness']['num_true_negatives'] == benchmark.num_true_negatives()


def test_num_true_positives():
    """Tests that the RAI num_true_positives calculation is correct."""
    assert metrics['prediction_fairness']['num_true_positives'] == benchmark.num_true_positives()


def test_positive_predictive_value():
    """Tests that the RAI positive_predictive_value calculation is correct."""
    assert metrics['prediction_fairness']['positive_predictive_value'] == benchmark.positive_predictive_value()


def test_smoothed_empirical_differential_fairness():
    """Tests that the RAI smoothed_empirical_differential_fairness calculation is correct."""
    assert metrics['prediction_fairness']['smoothed_empirical_differential_fairness'] == benchmark.smoothed_empirical_differential_fairness()


def test_true_negative_rate():
    """Tests that the RAI true_negative_rate calculation is correct."""
    assert metrics['prediction_fairness']['true_negative_rate'] == benchmark.true_negative_rate()


def test_true_positive_rate():
    """Tests that the RAI true_positive_rate calculation is correct."""
    assert metrics['prediction_fairness']['true_positive_rate'] == benchmark.true_positive_rate()


def test_true_positive_rate_difference():
    """Tests that the RAI true_positive_rate_difference calculation is correct."""
    assert metrics['prediction_fairness']['true_positive_rate_difference'] == benchmark.true_positive_rate_difference()


def test_true_positive_rate_difference():
    """Tests that the RAI true_positive_rate_difference calculation is correct."""
    assert metrics['prediction_fairness']['true_positive_rate_difference'] == benchmark.true_positive_rate_difference()
