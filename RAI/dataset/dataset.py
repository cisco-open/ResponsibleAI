# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from RAI.all_types import all_data_types
from abc import ABC

__all__ = ['Feature', 'MetaDatabase', 'Data', 'NumpyData', 'IteratorData', 'Dataset']


class Feature:
    """
    The RAI Feature Class represents a particular feature in the Dataset.
    It accepts a name, a data type, a description, whether the feature is categorical and its possible values.
    """

    def __init__(self, name: str, dtype: str, description: str, categorical=False, values=None) -> None:
        self.name = name
        self.dtype = dtype.lower()
        assert self.dtype in all_data_types, "dtype must be one of: " + str(all_data_types)
        self.description = description
        self.categorical = categorical
        self.values = values
        self.possibleValues = None
        if values is not None:
            self.possibleValues = values

    def __repr__(self) -> str:
        return f"{self.name}:{self.dtype}"


class Data(ABC):
    pass


class IteratorData(Data):
    def __init__(self, iterator, contains_x=True, contains_y=True):
        self.iterator = iterator
        self.iter = None
        self.contains_x = contains_x
        self.contains_y = contains_y
        self.X = None
        self.y = None
        self.rawX = None
        self.rawY = None
        self.mapping = {}
        self.categorical = None
        self.scalar = None
        self.image = None
        self.get_index = False

    def initialize(self, maps):
        self.mapping = maps

    def next_batch(self):
        val = next(self.iter, False)
        if not val:
            return val
        pos = 0
        x, y = None, None
        if self.contains_x:
            x = val[pos]
            self.rawX = x
            pos += 1
        if self.contains_y:
            y = val[pos]
            self.rawY = y
            self.y = y

        x, y = self._convert_image_data(x, y)
        self.X = x
        self.y = y
        self._separate_data(x)
        return val

    def reset(self):
        self.iter = iter(self.iterator)
        self.X = None
        self.y = None
        self.rawX = None
        self.categorical = None
        self.scalar = None
        self.image = None
        self.get_index = False

    def _convert_image_data(self, x, y):
        if x is not None:
            x = x.detach().numpy()
            x_shape = list(x.shape)
            x_shape[0] = 1
            x_shape.insert(0, -1)
            x = x.reshape(x_shape)
        if y is not None:
            y = y.detach().numpy()
        return x, y

    def _separate_data(self, x):
        if x is not None:
            self.scalar = None
            self.categorical = None
            self.image = None
            self.text = None
            if 'scalar' in self.mapping and any(val for val in self.mapping['scalar']):
                self.scalar = np.array(x[:, self.mapping['scalar']]).astype(np.float64)
            if 'categorical' in self.mapping and any(val for val in self.mapping['categorical']):
                self.categorical = np.array(x[:, self.mapping['categorical']]).astype(np.float64)
            if 'image' in self.mapping and any(val for val in self.mapping['image']):
                self.image = np.array(x[:, self.mapping['image']]).astype(np.float64)
            if 'text' in self.mapping and any(val for val in self.mapping['text']):
                self.text = np.array(x[:, self.mapping['text']]).astype(np.float64)


class NumpyData(Data):
    """
    The RAI Data class contains X and y data for a single Data split (train, test, val).
    """
    def __init__(self, X=None, y=None, rawX=None) -> None:
        self.X = X
        self.y = y
        self.contains_x = X is not None
        self.contains_y = y is not None
        self.rawX = rawX
        if self.rawX is None:
            self.rawX = self.X
        self.categorical = None
        self.scalar = None
        self.image = None
        self.text = None
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
    def initialize(self, masks):
        self.scalar = None
        self.categorical = None
        self.image = None
        self.text = None
        if self.X is not None:
            if "scalar" in masks and any(val for val in masks["scalar"]):
                self.scalar = np.array(self.X[:, masks["scalar"]]).astype(np.float64)
            if "categorical" in masks and any(val for val in masks["categorical"]):
                self.categorical = self.X[:, masks["categorical"]]
            if "image" in masks and any(val for val in masks["image"]):
                self.image = self.X[:, masks["image"]]
            if "text" in masks and any(val for val in masks["text"]):
                self.text = self.X[:, masks["scalar"]]


class Dataset:
    """
    The RAI Dataset class holds a dictionary of RAI Data classes,
    for example {'train': trainData, 'test': testData}, where trainData and testData
    are RAI Data objects.
    """

    def __init__(self, data_dict) -> None:
        self.data_dict = data_dict

    def separate_data(self, masks):
        for data in self.data_dict:
            self.data_dict[data].initialize(masks)


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
        self.text_mask = np.zeros(len(features), dtype=bool)
        self.scalar_map = []
        self.categorical_map = []
        self.image_map = []
        self.text_map = []
        self.data_format = set()
        self.stored_data = set()
        self.sensitive_features = []

        # Initialize maps and masks
        for i, f in enumerate(features):
            if f.dtype == "numeric":
                self.numerical_mask[i] = True
                if not f.categorical:
                    self.scalar_mask[i] = True
                    self.scalar_map.append(i)
                else:
                    self.categorical_map.append(i)
                    self.categorical_mask[i] = True
            elif f.dtype == "image":
                self.image_mask[i] = True
                self.image_map.append(i)
            elif f.dtype == "text":
                self.text_mask[i] = True
                self.text_map.append(i)
            elif f.dtype not in all_data_types:
                assert "Feature datatype must be one of: ", all_data_types

    def __repr__(self) -> str:
        return f" features: {self.features}"

    def initialize_requirements(self, dataset: Dataset, sensitive: bool) -> None:
        data_dict = dataset.data_dict
        if len(data_dict) > 0:
            data = list(dataset.data_dict.values())[0]
            if data.contains_x:
                self.stored_data.add("X")
            if data.contains_y:
                self.stored_data.add("y")
        if sensitive:
            self.stored_data.add("sensitive_features")
        for feature in self.features:
            self.data_format.add(feature.dtype)
