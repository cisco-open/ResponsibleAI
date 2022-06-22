import aif360.datasets.structured_dataset
import pytest
from numpy.testing import assert_almost_equal
import sklearn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAI.dataset import Data, Dataset
from RAI.AISystem import AISystem, Model
from RAI.utils import df_to_RAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

use_dashboard = False
np.random.seed(21)

# Get Dataset
data_path = "../data/adult/"
train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")

all_data = pd.concat([train_data, test_data], ignore_index=True)
idx = all_data['race'] != 'White'
all_data['race'][idx] = 'Black'

print(all_data.columns)
print(type(all_data))

meta, X, y = df_to_RAI(all_data, target_column="income-per-year", normalize=None, max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)

model = Model(agent=clf, task='binary_classification', description="Detect Cancer in patients using skin measurements",
              model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
ai = AISystem("AdultDB_Test1", meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)


names = [feature.name for feature in ai.meta_database.features]
df = pd.DataFrame(xTest, columns=names)
df['y'] = yTest

# structuredDataset = StructuredDataset(df, names, protected_attribute_names=['race'])
binDataset = BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=['race'])

print("type: ", type(binDataset))
benchmark = BinaryLabelDatasetMetric(binDataset)


clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
ai.compute({"test": predictions}, tag="Random Forest")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_base_rate():
    """Tests that the RAI pearson correlation calculation is correct."""
    assert metrics['dataset_fairness']['base-rate'] == benchmark.base_rate()


def test_num_instances():
    """Tests that the RAI pearson correlation calculation is correct."""
    assert metrics['dataset_fairness']['num-instances'] == benchmark.num_instances()


def test_num_negatives():
    """Tests that the RAI num negatives calculation is correct."""
    assert metrics['dataset_fairness']['num-negatives'] == benchmark.num_negatives()


def test_num_positives():
    """Tests that the RAI num positives calculation is correct."""
    assert metrics['dataset_fairness']['num-positives'] == benchmark.num_positives()
