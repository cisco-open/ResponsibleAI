import numpy as np
from RAI.all_types import all_data_types, all_data_types_lax

__all__ = ['Feature', 'MetaDatabase', 'Data', 'Dataset']


class Feature:
    """
    The RAI Feature Class represents a particular feature in the Dataset.
    It accepts a name, a data type, a description, whether the feature is categorical and its possible values.
    """

    def __init__(self, name: str, dtype: str, description: str, categorical=False, values=None) -> None:
        self.name = name
        self.dtype = dtype
        if dtype not in all_data_types_lax and not dtype.startswith("float") and not dtype.startswith("integer"):
            assert "dtype must be one of: ", all_data_types_lax
        self.description = description
        self.categorical = categorical
        self.values = values
        self.possibleValues = None
        if values is not None:
            self.possibleValues = values

    def __repr__(self) -> str:
        return f"{self.name}:{self.dtype}"


class Data:
    """
    The RAI Data class contains X and y data for a single Data split (train, test, val).
    """

    def __init__(self, X, y, rawX=None) -> None:
        self.X = X
        self.y = y
        self.rawX = rawX 
        self.categorical = None
        self.scalar = None
        self.image = None
        self.get_index = False

    def __len__(self):
        shape = np.shape(self.X)
        if len(shape) == 1:
            return 1
        return shape[0]

    def __getitem__(self, key):
        shape = np.shape(self.X)
        if len(shape) == 1:
            if key != 0:
                raise IndexError()
            if self.get_index:
                return self.X, self.Y, key 
            return self.X, self.Y
        if self.get_index:
            if self.y is None:
                return self.X[key], None, key
            else:
                return self.X[key], self.y[key], key

        if self.y is None:
            return self.X[key], None,
        else:
            return self.X[key], self.y[key]

    def getRawItem(self, key):
        return self.rawX[key]

    # Splits up a dataset into its different data types, currently scalar and categorical
    def separate(self, scalar_mask, categorical_mask, image_mask):
        self.scalar = self.X[:, scalar_mask]
        self.categorical = self.X[:, categorical_mask]
        self.image = self.X[:, image_mask]


class Dataset:
    """
    The RAI Dataset class holds a dictionary of RAI Data classes,
    for example {'train': trainData, 'test': testData}, where trainData and testData
    are RAI Data objects.
    """

    def __init__(self, data_dict, target=None, y_name=None) -> None:
        self.data_dict = data_dict
        self.target = target
        self.y_name = y_name

    def separate_data(self, scalar_mask, categorical_mask, image_mask):
        for data in self.data_dict:
            self.data_dict[data].separate(scalar_mask, categorical_mask, image_mask)


class MetaDatabase:
    """
    The RAI MetaDatabase class holds Meta information about the Dataset.
    It includes information about the features, and contains maps and masks to quick get access
    to the different feature data of different information.
    """

    def __init__(self, features) -> None:
        self.features = features
        self.scalar_mask = np.zeros(len(features), dtype=bool)
        self.categorical_mask = np.zeros(len(features), dtype=bool)
        self.numerical_mask = np.zeros(len(features), dtype=bool)
        self.image_mask = np.zeros(len(features), dtype=bool)
        self.scalar_map = []
        self.categorical_map = []
        self.image_map = []
        self.data_format = set()
        self.stored_data = set()
        self.sensitive_features = []

        # Initialize maps and masks
        for i, f in enumerate(features):
            if f.dtype.startswith("int") or f.dtype.startswith("float") or f.dtype == "Numeric":
                self.numerical_mask[i] = True
                if not f.categorical:
                    self.scalar_mask[i] = True
                    self.scalar_map.append(i)
                else:
                    self.categorical_map.append(i)
                    self.categorical_mask[i] = True
            elif f.dtype == "Image":
                self.image_mask[i] = True
                self.image_map.append(i)
            elif f.dtype not in all_data_types:
                assert "Feature datatype must be one of: ", all_data_types

    def __repr__(self) -> str:
        return f" features: {self.features}"

    def initialize_requirements(self, data: Data, sensitive: bool) -> None:
        if data.X is not None and len(data.X) != 0:
            self.stored_data.add("X")
        if data.y is not None and len(data.y) != 0:
            self.stored_data.add("y")
        if sensitive:
            self.stored_data.add("sensitive_features")
        for feature in self.features:
            if feature.dtype.startswith("float") or feature.dtype.startswith("int"):
                self.data_format.add("Numeric")
            else:
                self.data_format.add(feature.dtype)
