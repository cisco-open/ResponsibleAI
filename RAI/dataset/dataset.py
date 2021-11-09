# Dataset Requires Metrics as well 

__all__ = ['Feature', 'MetaDatabase', 'Data', 'Dataset']
import numpy as np


class Feature:
    def __init__(self, name, dtype, description) -> None:
        self.name = name
        self.dtype = dtype
        self.description = description


class MetaDatabase:
    def __init__(self, features) -> None:
        self.features = features

        
class Data:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    
    def __len__(self):
        shape = np.shape(self.X)
        if len(shape)==1:
            return 1
        return shape[0]
    

    def __getitem__(self, key):              
        shape = np.shape(self.X)
        if len(shape)==1:
            if key!=0:
                raise IndexError()
            return self.X,self.Y
        
        if self.y is None:
            return self.X[key],None
        else:
            return self.X[key],self.y[key]

         
class Dataset:
    def __init__(self, train_data, val_data=None, test_data=None) -> None:
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

