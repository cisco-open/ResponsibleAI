__all__=['TorchRaiDB']
import torch
import pandas as pd
import numpy as np


class TorchRaiDB( torch.utils.data.Dataset ):
     

    def __init__(self,X,y=None,meta = None, transform=None):
        """
        Args:
            X (dataframe): pandas dataframe
            y : optional target column
        """
        
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
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

