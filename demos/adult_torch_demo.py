
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
from sklearn.model_selection import train_test_split

data_path = "./data/adult/"


# this is the simple feedforward model implemented in torch
class Net(nn.Module):
    def __init__(self, input_size=30, scale=4):
        super().__init__()
        self.ff = nn.Sequential(*[  
            nn.Linear(input_size, 10*scale),
            nn.ReLU(),
            # nn.Linear(10*scale, 10*scale),
            # nn.ReLU(),
            nn.Linear(10*scale, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        return self.ff(x)


 
  
def main():

     
    train_data = pd.read_csv(data_path+"train.csv", header=0,
                    skipinitialspace=True, na_values="?")
    test_data = pd.read_csv(data_path+"test.csv", header=0,
                    skipinitialspace=True, na_values="?")
    all_data = pd.concat( [train_data, test_data],ignore_index=True)
    
    #convert aggregated data into RAI format
    meta, X,y  = df_to_RAI(all_data, target_column = "income-per-year", normalize="Scalar")
    
    #split train/test
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
    
    #temporarily reduce the computation 
    # xTest=xTest[:200,:]
    # yTest = yTest[:200]
     
    # create train and test data loader
    trdb = TorchRaiDB(xTrain,yTrain,meta)
    trdl = torch.utils.data.DataLoader(trdb, batch_size=20,shuffle=True, num_workers=4)
    
    tedb = TorchRaiDB(xTest,yTest,meta)
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
    
    
    dataset = Dataset(  train_data = Data(xTrain , yTrain), 
                        test_data = Data(xTest , yTest)) 
    ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
    ai.initialize()
    r = RaiRedis( ai )
    r.connect()
    r.reset_redis()
    # r.viewGUI()


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
    

      

    for ep in range(10):
        print( f"loss for epoch {ep}",  train_loop(20))
        preds=predict()

      
        ai.compute( preds , data_type="test")
        # print( ep, ai.get_metric_values())
        r.add_measurement( f"ep_{ep:02d}")

 
if __name__ == '__main__':
    
    main()
     
# Train model
