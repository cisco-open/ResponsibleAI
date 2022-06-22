import sklearn
import os
import sys
from RAI.dataset import Data, Dataset
from RAI.AISystem import AISystem, Model, Task
from RAI.utils import df_to_RAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
model = Model(agent=clf, task='binary_classification', description="Detect Cancer in patients using skin measurements",
              model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
ai = AISystem("AdultDB_Test1", meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
ai.compute(predictions, data_type="test", tag="Random Forest")

metrics = ai.get_metric_values()
info = ai.get_metric_info()

for g in metrics:
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])


def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (xTest == ai.dataset.test_data.X).all()
    assert (yTest == ai.dataset.test_data.y).all()
    assert (xTrain == ai.dataset.train_data.X).all()
    assert (yTrain == ai.dataset.train_data.y).all()


def test_accuracy():
    """Tests that the accuracy is correct."""
    assert metrics['performance_cl']['accuracy'] == sklearn.metrics.accuracy_score(yTest, predictions)


def test_auc():
    """Tests that the RAI auc function is correct."""
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(yTest, predictions)
    assert metrics['performance_cl']['auc'] == sklearn.metrics.auc(fpr, tpr)


def test_balanced_accuracy():
    """Tests that the RAI balanced accuracy calculation is correct."""
    assert metrics['performance_cl']['balanced_accuracy'] == \
           sklearn.metrics.balanced_accuracy_score(yTest, predictions)


# TODO: Should we be using macro f1?
def test_f1_score():
    """Tests that the RAI f1 score calculation is correct."""
    assert metrics['performance_cl']['f1_avg'] == sklearn.metrics.f1_score(yTest, predictions, average='macro')


def test_fp_rate():
    """Tests that the RAI f1 score calculation is correct."""
    assert metrics['performance_cl']['f1_avg'] == sklearn.metrics.f1_score(yTest, predictions, average='macro')


# TODO: Should we be using macro precision score?
def test_precision_score():
    """Tests that the RAI precision score calculation is correct."""
    assert metrics['performance_cl']['precision_score_avg'] == sklearn.metrics.precision_score(yTest, predictions, average='macro')


# TODO: Should fix metric calculation
def test_fp_rate():
    """Tests that the RAI fp rate calculation is correct."""
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(yTest, predictions).ravel()
    assert metrics['performance_cl']['fp_rate_avg'] == fp/(fp+tn)


def test_tp_rate():
    """Tests that the RAI tp rate calculation is correct."""
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(yTest, predictions).ravel()
    assert metrics['performance_cl']['tp_rate_avg'] == tp / (tp + fn)


# TODO: Recall score is TPR, remove TPR.
# TODO: Do we want Macro?
def test_recall_score():
    """Tests that the RAI recall score calculation is correct."""
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(yTest, predictions).ravel()
    assert metrics['performance_cl']['recall_score_avg'] == sklearn.metrics.recall_score(yTest, predictions, average='macro')


# TODO: Do we want macro?
def test_jaccard_score():
    """Tests that the RAI jaccard score calculation is correct."""
    result = sklearn.metrics.jaccard_score(yTest, predictions)
    print("TESTTEST, ", result)
    assert metrics['performance_cl']['jaccard_score_avg'] == sklearn.metrics.jaccard_score(yTest, predictions, average='macro')


def test_confusion_matrix():
    """Tests that the RAI confusion matrix calculation is correct."""
    assert (metrics['performance_cl']['confusion_matrix'] == sklearn.metrics.confusion_matrix(yTest, predictions)).all()
