# Dataset Requires Metrics as well

__all__ = ['Feature', 'MetaDatabase', 'Data', 'Dataset']
import numpy as np


class Feature:
    def __init__(self, name, dtype, description, categorical=False, values=None) -> None:
        self.name = name
        self.dtype = dtype
        self.description = description
        self.categorical = categorical
        self.values = values
        self.possibleValues = None
        if values is not None:
            self.possibleValues = values
            '''
            print("values: ", values)
            for value in values:
                self.possibleValues.append(value)
            print("Possible values: ", self.possibleValues)
            '''
    def __repr__(self) -> str:
        return f"{self.name}:{self.dtype}"


class MetaDatabase:
    def __init__(self, features) -> None:
        self.features = features
        self.scalar_mask = np.ones(len(features), dtype=bool)
        self.scalar_map = []
        self.categorical_map = []
        for i, f in enumerate(features):
            self.scalar_mask[i] = not f.categorical
            if f.categorical:
                self.scalar_map.append(i)
            else:
                self.categorical_map.append(i)

    def __repr__(self) -> str:
        return f" features: {self.features}"


class Data:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.categorical = None
        self.scalar = None
    
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
            return self.X[key], None
        else:
            return self.X[key], self.y[key]

    def separate(self, scalar_mask):
        self.scalar = self.X[:, scalar_mask]
        self.categorical = self.X[:, np.invert(scalar_mask)]


class Dataset:
    def __init__(self, train_data, val_data=None, test_data=None) -> None:
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def separate_data(self, scalar_mask):
        if self.train_data is not None:
            self.train_data.separate(scalar_mask)
        if self.val_data is not None:
            self.val_data.separate(scalar_mask)
        if self.test_data is not None:
            self.test_data.separate(scalar_mask)
