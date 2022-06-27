import sklearn
import os
import sys
from RAI.dataset import Data, Dataset
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

meta, X, y = df_to_RAI(all_data, target_column="income-per-year", normalize=None, max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

model = Model(agent=clf, name="Income classifier", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              model_class="Random Forest Classifier",)
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
ai = AISystem("AdultDB_Test1", meta_database=meta, task='binary_classification', dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
ai.compute({"test": {"predict": predictions}}, tag="Random Forest")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_inaccuracy():
    """Tests that the RAI inaccuracy calculation is correct."""
    assert metrics['adversarial_robustness']['inaccuracy'] == np.sqrt(1 - sklearn.metrics.accuracy_score(yTest, predictions))


def test_data_dimensionality():
    """Tests that the RAI data dimensionality calculation is correct."""
    assert metrics['adversarial_robustness']['data_dimensionality'] == np.sqrt(len(xTest[0]))
