#%%
# import imp
import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
 
import numpy as np
import torch

import pandas as pd

from sklearn.model_selection import train_test_split
data_path = "./data/adult/"

import numpy as np
# import RAI.utils


train_data = pd.read_csv(data_path+"train.csv", header=0,
                skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path+"test.csv", header=0,
                skipinitialspace=True, na_values="?")
all_data = pd.concat( [train_data, test_data],ignore_index=True)

# for data in [all_data]:
#     for i in all_data:
#         data[i].replace('nan', np.nan, inplace=True)
#         data[i].replace('?', np.nan, inplace=True)
#     data.dropna(inplace=True)
# print( all_data.isna().values.any())
# # Get Dataset

# features = []

# cat_columns = []
# for c in all_data.columns:
#     if all_data.dtypes[c] == "object" and c not in["income-per-year", "race","sex" ]:
#         cat_columns.append(c)

# all_data  = pd.get_dummies(all_data, columns = cat_columns, prefix = cat_columns)


# for c in all_data.columns:
#     if all_data.dtypes[c]=="float32":
#         features.append(Feature(c, "float32", c))
    


# features.append(Feature("race", "integer", "race value", categorical=True, 
#     values= {i:v for i,v in enumerate(all_data["race"].factorize()[1])}))
# features.append(Feature("sex", "integer", "sex value", categorical=True, 
#     values= {i:v for i,v in enumerate(all_data["sex"].factorize()[1])}))
# all_data["income-per-year"] = all_data["income-per-year"].factorize()[0]
# all_data["sex"] = all_data["sex"].factorize()[0]
# all_data["race"] = all_data["race"].factorize()[0]


# y = all_data.pop("income-per-year")
# X = all_data

from torch import nn
import torch   
from RAI.utils.utils import df_to_RAI
class Net(nn.Module):
    def __init__(self, input_size=30, scale=10):
        super().__init__()
        self.ff = nn.Sequential([  
            nn.Linear(input_size, 30*scale),
            nn.ReLU(),
            nn.Linear(30*scale, 30*scale),
            nn.ReLU(),
            nn.Linear(20*scale, 20*scale),
            nn.ReLU(),
            nn.Linear(30*scale, 1),
        
        ])
        
    def forward(self, x):
        return self.ff(x)
class TorchDF( torch.utils.data.Dataset ):
     

    def __init__(self,X,y=None,meta = None, transform=None):
        """
        Args:
            X (dataframe): pandas dataframe
            y : optional target column
        """
        self.X = X
        self.y = y
        self.meta = meta 
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):


        if not self.meta:
            if self.y:
                return self.X.iloc[idx,:].to_numpy(dtype="float32"), self.y[idx]
            else:
                return self.Xiloc[idx,:].to_numpy(dtype="float32")

        X =  []

        def onehot(x,n):
            res = [0]*n
            res[x]=1
            return res
        
        for i,f in enumerate(self.meta.features):
            if f.categorical:
                X.extend( onehot(self.X.iloc[idx,i] , len(f.values)  )   )
            else:
                X.append( self.X.iloc[idx,i])

        if self.y is not None:
            return np.array(X,dtype="float"),self.y[idx]
        else:
            return np.array(X,dtype="float") 

def main():


    meta, X,y  = df_to_RAI(all_data, target_column = "income-per-year")

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
    

    trdb = TorchDF(xTrain,yTrain,meta)
    trdl = torch.utils.data.DataLoader(trdb, batch_size=4,shuffle=True, num_workers=4)
    for x,y in trdl:
        print(x.shape,y)
        exit(0)
 

 
# device = "cpu"
# net  = Net( xTest.shape[1]).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
# model = Model(agent=net, name="Adult", display_name="Adult Income Prediction Task",
#                   model_class="Neural Network", adaptive=True,
#                   optimizer=optimizer, loss_function=criterion)
# task = Task(model=model, type='binary_classification',
#                 description="predict if income above 50K")
# configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
#                               "protected_attributes": ["race"], "positive_label": 1},
#                  "time_complexity": "polynomial"}
# ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
# ai.initialize()
# ai.reset_redis()

#%%
# Hook data in with our Representation
 
# Create AISystem from previous objects. AISystems are what users will primarily interact with.

# ai.initialize()
if __name__ == '__main__':
    main()
# Train model
