__all__ = ['Feature', 'MetaDatabase', 'Data', 'Dataset']
import numpy as np


all_data_types = {"numeric", "image", "text"}
all_data_types_lax = {"integer", "float", "numeric", "image", "text"}

class Feature:
    def __init__(self, name: str, dtype: str, description: str, categorical=False, values=None, sensitive=False) -> None:
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
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.categorical = None
        self.scalar = None

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
            return self.X, self.Y

        if self.y is None:
            return self.X[key], None
        else:
            return self.X[key], self.y[key]

    def separate(self, scalar_mask):
        self.scalar = self.X[:, scalar_mask]
        self.categorical = self.X[:, np.invert(scalar_mask)]


class Dataset:
    def __init__(self, data_dict) -> None:
        self.data_dict = data_dict

    def separate_data(self, scalar_mask):
        for data in self.data_dict:
            self.data_dict[data].separate(scalar_mask)


class MetaDatabase:
    def __init__(self, features) -> None:
        self.features = features
        self.scalar_mask = np.ones(len(features), dtype=bool)
        self.numerical_mask = np.zeros(len(features), dtype=bool)
        self.scalar_map = []
        self.categorical_map = []
        self.data_format = set()
        self.stored_data = set()
        self.sensitive_features = []

        for i, f in enumerate(features):
            self.scalar_mask[i] = not f.categorical
            # TODO: Once images are added, this will also need incorporate images
            if f.dtype.startswith("int") or f.dtype.startswith("float") or f.dtype == "numeric":
                self.numerical_mask[i] = True
                if not f.categorical:
                    self.scalar_map.append(i)
                else:
                    self.categorical_map.append(i)
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
                self.data_format.add("numeric")
            else:
                self.data_format.add(feature.dtype)