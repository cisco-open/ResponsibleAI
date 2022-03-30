#%%
 
import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
 
import numpy as np
import pandas as pd


import torch  
from torch import nn

from sklearn.model_selection import train_test_split
data_path = "./data/adult/"

import numpy as np
# import RAI.utils


train_data = pd.read_csv(data_path+"train.csv", header=0,
                skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path+"test.csv", header=0,
                skipinitialspace=True, na_values="?")
all_data = pd.concat( [train_data, test_data],ignore_index=True)
 

 
from RAI.utils.utils import df_to_RAI
class Net(nn.Module):
    def __init__(self, input_size=30, scale=10):
        super().__init__()
        self.ff = nn.Sequential(*[  
            nn.Linear(input_size, 10*scale),
            nn.ReLU(),
            nn.Linear(10*scale, 10*scale),
            nn.ReLU(),
            nn.Linear(10*scale, 1),
            nn.Sigmoid()
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
                return self.X.iloc[idx,:].to_numpy(dtype="float32"), self.y[idx].astype("float32")
            else:
                return self.Xiloc[idx,:].to_numpy(dtype="float32")

        X =  []

        def onehot(x,n):
            res = [0]*n
            res[int(x)]=1
            return res
        
        for i,f in enumerate(self.meta.features):
            if f.categorical:
                X.extend( onehot(self.X[idx,i] , len(f.values)  )   )
            else:
                X.append( self.X[idx,i])

        if self.y is not None:
            return np.array(X,dtype="float32"),self.y[idx].astype("float32")
        else:
            return np.array(X,dtype="float32") 


class TorchRaiDB( torch.utils.data.Dataset ):
     

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
                return self.X[idx,:], self.y[idx]
            else:
                return self.X[idx,:]

        X =  []

        def onehot(x,n):
            res = [0]*n
            res[x]=1
            return res
        
        for i,f in enumerate(self.meta.features):
            if f.categorical:
                X.extend( onehot(self.X[idx,i] , len(f.values)  )   )
            else:
                X.append( self.X[idx,i])

        if self.y is not None:
            return np.array(X,dtype="float32"),self.y[idx].astype("float32")
        else:
            return np.array(X,dtype="float32") 

def main():


    meta, X,y  = df_to_RAI(all_data, target_column = "income-per-year", normalize="Scalar")
    
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
    
    


    xTest=xTest[:200,:]
    yTest = yTest[:200]
    
    training_data = Data(xTrain , yTrain)  # Accepts Data and GT
    test_data = Data(xTest , yTest)
    dataset = Dataset(training_data, test_data=test_data) 

    trdb = TorchDF(xTrain,yTrain,meta)
    trdl = torch.utils.data.DataLoader(trdb, batch_size=20,shuffle=True, num_workers=4)
    
    tedb = TorchDF(xTest,yTest,meta)
    tedl = torch.utils.data.DataLoader(tedb, batch_size=20,shuffle=False, num_workers=4)
    
    
 
 

 
    device = "cpu"
    # xTrain dim is different than the tensor dimention since we need to convert categortical to onehot
    input_dim = next(iter(trdl))[0].shape[1]
    net  = Net(input_dim).to(device)
    criterion = nn.BCELoss().to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    model = Model(agent=net, name="Adult", display_name="Adult Income Prediction Task",
                    model_class="Neural Network", adaptive=True,
                    optimizer=optimizer, loss_function=criterion)
    task = Task(model=model, type='binary_classification',
                    description="predict if income above 50K")
    configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                                "protected_attributes": ["race"], "positive_label": 1},
                    "time_complexity": "polynomial"}
    ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
    ai.initialize()
    ai.reset_redis()


    def train_loop(max_iter=None):
        running_loss = 0
        count = 0
        for x,y in trdl:
            optimizer.zero_grad()
            yhat = net(x)
            loss = criterion(yhat[:,0], y)
            loss.backward()

            optimizer.step( )
            count+=1
            running_loss += loss.item()
            if max_iter and count>max_iter:
                break
        return running_loss/count

    def predict():
        
        res = []
        count = 0
        with torch.no_grad():
            for x,y in tedl:
                yhat = net(x)
                res.append( yhat.detach().numpy()) 
        return (np.concatenate(res,0)>.5).astype("float32")[:,0]
    ai.viewGUI()


    import traceback
    import warnings
    import sys

    # def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    #     log = file if hasattr(file,'write') else sys.stderr
    #     traceback.print_stack(file=log)
    #     log.write(warnings.formatwarning(message, category, filename, lineno, line))

    # warnings.showwarning = warn_with_traceback

    for ep in range(1,21):
        print( "loss for epoch %d"%ep, train_loop(100))
        ai.compute_metrics( preds = predict(), data_type="test", export_title= "epoch_%d"%ep)
         
        # print( ep, metrics)



#%%
# Hook data in with our Representation
 
# Create AISystem from previous objects. AISystems are what users will primarily interact with.

# ai.initialize()
if __name__ == '__main__':
    main()
# Train model
