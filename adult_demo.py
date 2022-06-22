import pandas as pd
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import Data, Dataset
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI

use_dashboard = True

# Get Dataset
data_path = "./data/adult/"

train_data = pd.read_csv(data_path + "train.csv", header=0,
                         skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0,
                        skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)

# convert aggregated data into RAI format
meta, X, y = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar", max_categorical_threshold=5)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

# Create a model to make predictions
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

model = Model(agent=reg, task='binary_classification', description="Detect Cancer in patients using skin measurements",
              model_class="Random Forest Classifier", )
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
ai = AISystem(name="AdultDB", meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)

print("\n\nTESTING PREDICTING METRICS:")
test_preds = reg.predict(xTest)
ai.compute({"test": reg.predict(xTest)}, tag='model1')

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
    r.add_measurement()
    # r.viewGUI()

reg2 = AdaBoostClassifier()
reg2.fit(xTrain, yTrain)
ai.set_agent(reg2)

ai.compute({"test": reg.predict(xTest)}, tag="model2")
v = ai.get_metric_values()
v = v["test"]
info = ai.get_metric_info()
if use_dashboard:
    r.add_measurement()

for g in v:
    for m in v[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, v[g][m])
