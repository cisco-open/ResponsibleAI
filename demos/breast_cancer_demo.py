import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAI.dataset import Data, Dataset
from RAI.AISystem import AISystem, Model, Task
from RAI.utils import df_to_RAI
from RAI.torch import TorchRaiDB
from RAI.redis import RaiRedis
import numpy as np
import pandas as pd
import torch  
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
use_dashboard  = False

# Get Dataset
df = load_breast_cancer(return_X_y = False, as_frame=True)
df = df.data.join( df.target )

meta, X,y  = df_to_RAI(df, target_column = "target", normalize="Scalar", max_categorical_threshold = 5)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
 
  
# Create a model to make predictions
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

model = Model(agent=reg, name="BreastCancer", display_name="sklearn breast cancer data", model_class="Random Forest Classifier", adaptive=False)
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = { }

dataset = Dataset(  train_data = Data(xTrain , yTrain), 
                        test_data = Data(xTest , yTest)) 
ai = AISystem(meta_database=meta, dataset=dataset, task=task)
ai.initialize( user_config=configuration)


reg.fit(xTrain, yTrain)
 
print("\n\nTESTING PREDICTING METRICS:")
test_preds = reg.predict(xTest)
ai.compute( reg.predict(xTest) , data_type="test")

if use_dashboard:
    r = RaiRedis( ai )
    r.connect()
    r.reset_redis()
    r.add_measurement( f"random_forest")
    r.viewGUI()

v = ai.get_metric_values()
info = ai.get_metric_info()

for g in v:
    print("\n\ngroup : ", g)
    for m in v[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"]in ("numeric","vector-dict","text"):
                print ('--------- ' ,g, m, v[g][m])
            
            else:
                print ('vvvvvvvvv ' ,g, m, type(v[g][m]), info[g][m]["type"])
        else:
            print(     '????????? ', g,m, v[g][m])
