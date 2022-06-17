import pytest
from numpy.testing import assert_almost_equal
import sklearn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAI.dataset import Data, Dataset
from art.metrics.verification_decisions_trees import RobustnessVerificationTreeModelsCliqueMethod
from RAI.AISystem import AISystem, Model, Task
from RAI.utils import df_to_RAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

tag = "Random Forest"
task_type = "binary_classification"
description = "Detect Cancer in patients using skin measurements"

clf = RandomForestClassifier()
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


def test_adversarial_tree():
    """Tests that the feature names are correct."""
    rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=clf, verbose=False)
    bound, error = rt.verify(xTrain, y, eps_init=0.3, nb_search_steps=2, max_clique=2, max_level=2)
    assert metrics['art-trees']['adversarial-tree-verification-bound'] == bound
    assert metrics['art-trees']['adversarial-tree-verification-error'] == error


# TODO: These tests run extremely slow on my laptop
# TODO: Generalize function across all sklearn trees:
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb