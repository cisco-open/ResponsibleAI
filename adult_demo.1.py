import os, sys
from RAI.dataset import Data, Dataset
from RAI.AISystem import AISystem, Model
from RAI.utils import df_to_RAI
from RAI.redis import RaiRedis
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
use_dashboard = True

# Get Dataset
data_path = "./data/adult/"

train_data = pd.read_csv(data_path+"train.csv", header=0,
                    skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path+"test.csv", header=0,
                skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data],ignore_index=True)

#convert aggregated data into RAI format
meta, X,y = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar", max_categorical_threshold=5)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
 
  
# Create a model to make predictions
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)

model = Model(agent=reg, task='binary_classification', predict_fun=reg.predict, predict_prob_fun=reg.predict_proba,
              description="Detect Cancer in patients using skin measurements", model_class="Random Forest Classifier")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                                "protected_attributes": ["race"], "positive_label": 1},
                    "time_complexity": "polynomial"}

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
ai = AISystem("AdultDB", meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)


if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()


def test_model(mdl, name):
    mdl.fit(xTrain, yTrain)
    ai.set_agent(mdl)
    ai.compute({"test": mdl.predict(xTest)}, tag=name)
 
    if use_dashboard:
        r.add_measurement()


test_model(reg, "mdl1")
reg2 = RandomForestClassifier(n_estimators=11, criterion='gini', random_state=0, min_samples_leaf=10, max_depth=3)
 
test_model(reg2, "mdl2")
# for g in v:
    
#     for m in v[g]:
#         if "type" in info[g][m]:
#             if info[g][m]["type"]in ("numeric","vector-dict", "text"):
#                 print (g,  m, v[g][m])
            
 