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


import sklearn
import os
import sys
from RAI.dataset import NumpyData, Dataset
from RAI.AISystem import AISystem, Model
from RAI.utils import df_to_RAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
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

model = Model(agent=clf, output_features=output, name="income ai", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})
ai = AISystem(name="AdultDB_Test1", task='binary_classification', meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
ai.compute({"test": {"predict": predictions}}, tag="Random Forest")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (xTest == ai.dataset.data_dict["test"].X).all()
    assert (yTest == ai.dataset.data_dict["test"].y).all()
    assert (xTrain == ai.dataset.data_dict["train"].X).all()
    assert (yTrain == ai.dataset.data_dict["train"].y).all()


def test_accuracy():
    """Tests that the accuracy is correct."""
    assert metrics['performance_cl']['accuracy'] == sklearn.metrics.accuracy_score(yTest, predictions)


def test_balanced_accuracy():
    """Tests that the RAI balanced accuracy calculation is correct."""
    assert metrics['performance_cl']['balanced_accuracy'] == \
           sklearn.metrics.balanced_accuracy_score(yTest, predictions)


# TODO: Macro?
def test_f1_score():
    """Tests that the RAI f1 score calculation is correct."""
    assert metrics['performance_cl']['f1_avg'] == sklearn.metrics.f1_score(yTest, predictions, average='macro')


def test_fp_rate():
    """Tests that the RAI f1 score calculation is correct."""
    assert metrics['performance_cl']['f1_avg'] == sklearn.metrics.f1_score(yTest, predictions, average='macro')


# TODO: Macro?
def test_precision_score():
    """Tests that the RAI precision score calculation is correct."""
    assert metrics['performance_cl']['precision_score_avg'] == sklearn.metrics.precision_score(yTest, predictions, average='macro')


def test_fp_rate():
    """Tests that the RAI fp rate calculation is correct."""
    confusion_matrix = sklearn.metrics.confusion_matrix(yTest, predictions)
    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    tn = confusion_matrix.sum() - fp - fn - tp
    assert metrics['performance_cl']['fp_rate_avg'] == np.average(fp / (fp + tn))


# TODO: Macro?
def test_recall_score():
    """Tests that the RAI recall score calculation is correct."""
    assert metrics['performance_cl']['recall_score_avg'] == sklearn.metrics.recall_score(yTest, predictions, average='macro')


# TODO: macro?
def test_jaccard_score():
    """Tests that the RAI jaccard score calculation is correct."""
    assert metrics['performance_cl']['jaccard_score_avg'] == sklearn.metrics.jaccard_score(yTest, predictions, average='macro')


def test_confusion_matrix():
    """Tests that the RAI confusion matrix calculation is correct."""
    assert (metrics['performance_cl']['confusion_matrix'] == sklearn.metrics.confusion_matrix(yTest, predictions)).all()
