import pytest
from numpy.testing import assert_almost_equal
import sklearn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAI.dataset import Data, Dataset
from RAI.AISystem import AISystem, Model, Task
from RAI.utils import df_to_RAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

tag = "Random Forest"
task_type = "binary_classification"
description = "Detect Cancer in patients using skin measurements"

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
use_dashboard = False
np.random.seed(21)

# %%
# Get Dataset
data_path = "../data/adult/"
train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")

all_data = pd.concat([train_data, test_data], ignore_index=True)
idx = all_data['race'] != 'White'
all_data['race'][idx] = 'Black'

# %%
# convert aggregated data into RAI format
meta, X, y = df_to_RAI(all_data, target_column="income-per-year", normalize=None, max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

# Create a model to make predictions
model = Model(agent=clf, model_class="Random Forest Classifier")
task = Task(model=model, type=task_type, description=description)
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset(train_data=Data(xTrain, yTrain),
                  test_data=Data(xTest, yTest))
ai = AISystem("AdultDB_Test1", meta_database=meta, dataset=dataset, task=task, enable_certificates=False)
ai.initialize(user_config=configuration)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
ai.compute(predictions, data_type="test", tag=tag)

metrics = ai.get_metric_values()
info = ai.get_metric_info()

for g in metrics:
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])

print(metrics['Tree Models'])


def test_feature_names():
    """Tests that the feature names are correct."""
    assert metrics['Tree Models']['feature_names'] == [f.name for f in ai.meta_database.features]


def test_estimator_counts():
    """Tests that the feature names are correct."""
    assert metrics['Tree Models']['estimator_counts'] == clf.n_estimators


def test_estimators():
    """Tests that the feature names are correct."""
    assert metrics['Tree Models']['estimator_params'] == clf.estimators_


# TODO change name to match the convention
