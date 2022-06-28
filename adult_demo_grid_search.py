import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import Data, Dataset
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
use_dashboard = True
data_path = "./data/adult/"

train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)

# convert aggregated data into RAI format
meta, X, y = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar", max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
model = Model(agent=clf, name="cisco_income_ai", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              description="Income Prediction AI", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train_data": Data(xTrain, yTrain), "test_data": Data(xTest, yTest)})
ai = AISystem("AdultDB_GridSearch", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiRedis(ai)
    r.reset_redis()


def test_model(mdl, name):
    mdl.fit(xTrain, yTrain)
    ai.set_agent(mdl)
    ai.compute({"test": {"predict", mdl.predict(xTest)}}, tag=name)

    if use_dashboard:
        r.add_measurement()


for n_s in [2, 5, 10, 20]:
    for max_depth in [1, 2, 3]:
        for criterion in ['entropy', 'gini']:
            name = f"n:{n_s}, d:{max_depth}, {criterion}"
            print("grid searching param :", name)
            mdl = RandomForestClassifier(n_estimators=n_s, criterion=criterion, random_state=0, min_samples_leaf=5, max_depth=max_depth)
            test_model(mdl, name)
