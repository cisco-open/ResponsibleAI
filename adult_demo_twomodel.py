import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import Data, Dataset
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
use_dashboard = True
np.random.seed(21)

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)

data_path = "./data/adult/"
train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)
idx = all_data['race'] != 'White'
all_data['race'][idx] = 'Black'
target_column="income-per-year"

# convert aggregated data into RAI format
# print(all_data.pop("income-per-year"))
meta, X, y, y_name = df_to_RAI(all_data, target_column=target_column, normalize=None, max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

# Create a model to make predictions
model = Model(agent=clf,  name="cisco_income_ai", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              description="Income Prediction", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)}, target_column, y_name)
ai = AISystem("AdultDB_3", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)
predictions_train = clf.predict(xTrain)
ai.compute({"test": {"predict": predictions}, "train": {"predict": predictions_train}}, tag="Random Forest 5 Estimator")


if use_dashboard:
    r.add_measurement()
    r.delete_data("AdultDB_2")
    r.delete_data("AdultDB_Test1")
    r.delete_data("AdultDB")


mdl = RandomForestClassifier(n_estimators=10, min_samples_leaf=20, max_depth=2)
mdl.fit(xTrain, yTrain)
predictions = mdl.predict(xTest)
predictions_train = mdl.predict(xTrain)
ai.compute({"test": {"predict": predictions}, "train": {"predict": predictions_train}}, tag="Random Forest 10 Estimator")
if use_dashboard:
    r.add_measurement()
    r.export_metadata()

ai.display_metric_values(display_detailed=True)
