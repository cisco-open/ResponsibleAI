#%%
# imports
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, min_samples_leaf=5, max_depth=2)
use_dashboard  = True

np.random.seed(21)


#%%
# Get Dataset
data_path = "./data/adult/"

train_data = pd.read_csv(data_path+"train.csv", header=0,
                    skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path+"test.csv", header=0,
                skipinitialspace=True, na_values="?")
all_data = pd.concat( [train_data, test_data],ignore_index=True)
idx = all_data['race']!='White'
all_data['race'][ idx ] ='Black'

#%%
#convert aggregated data into RAI format
meta, X,y  = df_to_RAI(all_data, target_column = "income-per-year", normalize = None, max_categorical_threshold = 5)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
 
  

#%%  
# Create a model to make predictions
model = Model(agent=clf, model_class="Random Forest Classifier")
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                                "protected_attributes": ["race"], "positive_label": 1},
                    "time_complexity": "polynomial"}

dataset = Dataset(  train_data = Data(xTrain , yTrain), 
                        test_data = Data(xTest , yTest)) 
ai = AISystem("AdultDB_Test1", meta_database=meta, dataset=dataset, task=task )
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiRedis( ai )
    r.reset_redis()


#%%
   
def test_model(mdl, name, sample_weight = None):
    
    mdl.fit(xTrain,yTrain, sample_weight = sample_weight)
    ai.set_agent( mdl )
    ai.compute( mdl.predict(xTest), data_type="test", tag=name)
 
    if use_dashboard:
        r.add_measurement()


#%%
#test random forest
# mdl = RandomForestClassifier(n_estimators=10, criterion='entropy',
#                              random_state=0, min_samples_leaf=5, max_depth=1)
# test_model( mdl, "Random Forest d1")


#%%
mdl = RandomForestClassifier(n_estimators=5, min_samples_leaf=20, max_depth=2)
test_model( mdl, "Random Forest")


# #%%
# #test Ada Boost
# mdl = AdaBoostClassifier(n_estimators=10  )
# test_model( mdl, "Ada Boost")
 
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

# xTrain = BinaryLabelDataset(favorable_label=1,
#                                 unfavorable_label=0,
#                                 df=xTrain,
#                                 label_names=['output'],
#                                 protected_attribute_names=['group'],
#                                 unprivileged_protected_attributes=['0'])

# idx = 8
# s =  pd.value_counts(xTrain[:,idx]  )
# f = s/sum(s)
# print(f)
# w = f[ xTrain[:,idx].astype(int)]
from imblearn.over_sampling import SMOTE
xTrain = np.hstack([xTrain,  yTrain[:,np.newaxis]])
xTrain, yTrain = SMOTE().fit_resample(xTrain, xTrain[:,-1])
yTrain = xTrain[:,-1].astype(int)
xTrain = xTrain[:,:-1]

mdl = RandomForestClassifier(n_estimators=5, min_samples_leaf=20, max_depth=2)
test_model( mdl, "Random Forest with Reweighting" )


#%%
metrics = ai.get_metric_values()
info = ai.get_metric_info()


# %%

for g in metrics:
    
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"]in ("numeric","vector-dict","text"):
                print (g, m, metrics[g][m])
            
       
# %%
