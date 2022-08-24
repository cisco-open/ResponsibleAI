import os
import sys
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import NumpyData, Dataset
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI
from sklearn.ensemble import RandomForestClassifier
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
use_dashboard = True
data_path = "../data/adult/"

train_data = pd.read_csv(data_path + "train.csv", header=0, skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0, skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)

# convert aggregated data into RAI format
meta, X, y, output = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar", max_categorical_threshold=5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
clf.fit(xTrain, yTrain)
model = Model(agent=clf, output_features=output, name="cisco_income_ai", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
              description="Income Prediction AI", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": NumpyData(xTrain, yTrain, xTrain), "test": NumpyData(xTest, yTest, xTest)})
ai = AISystem(name="AdultDB_GridSearch", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()


def test_model(mdl, name):
    mdl.fit(xTrain, yTrain)
    ai.model.agent = mdl
    ai.model.predict_fun = clf.predict
    ai.model.predict_prob_fun = clf.predict_proba
    ai.compute({"test": {"predict": mdl.predict(xTest)}}, tag=name)

    if use_dashboard:
        r.add_measurement()
        r.export_metadata()


for n_s in [2, 5, 10, 20]:
    for max_depth in [1, 2, 3]:
        for criterion in ['entropy', 'gini']:
            name = f"n:{n_s}, d:{max_depth}, {criterion}"
            print("grid searching param :", name)
            mdl = RandomForestClassifier(n_estimators=n_s, criterion=criterion, random_state=0, min_samples_leaf=5, max_depth=max_depth)
            test_model(mdl, name)

if use_dashboard:
    r.export_visualizations("test", "test")
