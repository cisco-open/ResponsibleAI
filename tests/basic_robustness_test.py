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
model = Model(agent=clf, name="Income classifier", task='binary_classification', predict_fun=clf.predict,
              predict_prob_fun=clf.predict_proba, model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
ai = AISystem("AdultDB_Test1", meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
ai.compute({"test": predictions}, tag="Random Forest")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


for g in metrics:
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])


def test_normalized_feature_std():
    """Tests that the RAI normalized_feature_std calculation is correct."""
    mean_v = np.mean(xTest, axis=0, keepdims=True)
    std_v = np.std(xTest, axis=0, keepdims=True)
    assert metrics['basic_robustness']['normalized_feature_std'] == bool(np.all(np.isclose(std_v, np.ones_like(std_v)))
                                                                         and np.all(np.isclose(mean_v, np.ones_like(mean_v))))


def test_normalized_feature_01():
    """Tests that the RAI normalized_feature_01 calculation is correct."""
    max_v = np.max(xTest, axis=0, keepdims=True)
    min_v = np.min(xTest, axis=0, keepdims=True)
    assert metrics['basic_robustness']['normalized_feature_01'] == bool(np.all(np.isclose(max_v, np.ones_like(max_v)))
                                                                         and np.all(np.isclose(min_v, np.zeros_like(min_v))))
