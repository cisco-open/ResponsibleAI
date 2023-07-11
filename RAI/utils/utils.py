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


import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from threading import Timer
from sklearn.preprocessing import StandardScaler
from RAI.dataset.dataset import Feature, MetaDatabase
import torchvision.transforms as transforms

__all__ = ['jsonify', 'compare_runtimes', 'df_to_meta_database', 'df_to_RAI', 'reweighing',
           'calculate_per_mapped_features', 'convert_float32_to_float64',
           'convert_to_feature_value_dict', 'convert_to_feature_dict', 'map_to_feature_array', 'map_to_feature_dict',
           'torch_to_RAI', 'modals_to_RAI']


# TODO: Remove?
def reweighing():
    pass


def is_primitive(obj):
    return not hasattr(obj, '__dict__')


def jsonify(v):
    if type(v) is np.ma.MaskedArray:
        return np.ma.getdata(v).tolist()
    if type(v) is np.ndarray:
        return clean_list(v.tolist())
    if type(v) is list:
        return clean_list(v)
    if type(v) in (np.bool, '_bool', 'bool_') or v.__class__.__name__ == "bool_":
        return bool(v)
    if (isinstance(v, int) or isinstance(v, float)) and (
            math.isinf(v) or math.isnan(v)):  # CURRENTLY REPLACING INF VALUES WITH NULL
        return None
    if is_primitive(v):
        return v
    return pickle.dumps(v).decode('ISO-8859-1')


# jsonifies each element of a list v
def clean_list(v):
    for i in range(len(v)):
        v[i] = jsonify(v[i])
    return v


# Returns True if the seen runtime is less than or equal to the required
def compare_runtimes(required, seen):
    required = complexity_to_integer(required)
    seen = complexity_to_integer(seen)
    return seen <= required


def complexity_to_integer(complexity):
    if type(complexity) is str:
        complexity = complexity.lower()
    result = 4
    if complexity == "linear":
        result = 1
    elif complexity == "polynomial":
        result = 2
    elif complexity == "exponential":
        result = 3
    return result


# Creates a RAI Metadatabase using meta information from a pandas dataframe
def df_to_meta_database(df, categorical_values=None, protected_attribute_names=None, privileged_info=None,
                        positive_label=None):
    features = []
    fairness_config = {}
    for col in df.columns:
        categorical = categorical_values is not None and col in categorical_values
        features.append(Feature(col, "float32", col, categorical=categorical, values=categorical_values.get(col, None)))
    if protected_attribute_names is not None:
        fairness_config["protected_attributes"] = protected_attribute_names
    if privileged_info is not None:
        fairness_config["priv_group"] = privileged_info
    if positive_label is not None:
        fairness_config["positive_label"] = positive_label
    meta = MetaDatabase(features)
    return meta, fairness_config


def df_remove_nans(df, extra_symbols):
    for i in df:
        df[i].replace('nan', np.nan, inplace=True)
        for s in extra_symbols:
            df[i].replace(s, np.nan, inplace=True)
    df.dropna(inplace=True)


def torch_to_RAI(torch_item, max_size=None, detailed=True):
    result_x = None
    result_y = []
    raw_x = None
    x_shape = None

    if isinstance(torch_item, torch.utils.data.DataLoader):
        transform = torch_item.dataset.transform
        if torch_item.dataset.transform is None:
            torch_item.dataset.transform = transforms.ToTensor()
            torch_item.transform = transform
        for i, val in enumerate(torch_item, 0):
            x, y = val
            if raw_x is None:
                raw_x = x
            else:
                raw_x = torch.vstack((raw_x, x))
            x = x.detach().numpy()
            y = y.detach().numpy()
            if x_shape is None:
                x_shape = list(x.shape)
                x_shape[0] = 1
                x_shape.insert(0, -1)
            x = x.reshape(x_shape)
            if result_x is None:
                result_x = x
            else:
                result_x = np.vstack((result_x, x))
            result_y.extend(y.tolist())
            if max_size is not None and len(result_x) > max_size:
                result_x = result_x[:max_size]
                break

    elif isinstance(torch_item, torch.Tensor):
        result_x = np.array([[x] for x in torch_item])
    else:
        assert "torch_item must be of type DataLoader or Tensor"
    return result_x, result_y, raw_x


# Converts a pandas dataframe to a Rai Metadatabase and X and y data.
def df_to_RAI(df, target_column=None, clear_nans=True, extra_symbols="?", normalize=None,
              max_categorical_threshold=None, text_columns=[]):
    if clear_nans:
        df_remove_nans(df, extra_symbols)
    if max_categorical_threshold:
        for col in df:
            if len(df[col].unique()) < max_categorical_threshold:
                df[col] = pd.Categorical(df[col])
    if normalize is not None:
        if normalize == "Scalar":
            num_d = df.select_dtypes(exclude=['object', 'category'])
            df[num_d.columns] = StandardScaler().fit_transform(num_d)
    output_feature = []
    if target_column:
        y = df.pop(target_column)
        categorical = str(y.dtypes) in ["object", "category"]
        if y.name in text_columns:
            f = Feature(y.name, "text", y.name)
            y = y.tolist()
        elif categorical:
            fact = y.factorize(sort=True)
            f = Feature(
                y.name, "numeric", y.name, categorical=True,
                values={i: v for i, v in enumerate(fact[1])}
            )
            y, _ = y.factorize(sort=True)
        else:
            f = Feature(y.name, "numeric", y.name)
            y = y.tolist()
        output_feature.append(f)
    else:
        y = None

    features = []

    for c in df:
        if c in text_columns:
            f = Feature(c, "text", c)
        elif str(df.dtypes[c]) in ["object", "category"]:
            fact = df[c].factorize(sort=True)
            df[c] = fact[0]
            f = Feature(
                c, "numeric", c, categorical=True,
                values={i: v for i, v in enumerate(fact[1])}
            )
        elif "float" in str(df.dtypes[c]):
            f = Feature(c, "numeric", c)
        features.append(f)
    return MetaDatabase(features), df.to_numpy(), y, output_feature


# Converts a pandas dataframe with numeric data and text,
# as well as image dictionaries to a Rai Metadatabase and X and y data.
# TODO: This needs to be formalized for multi modal data
def modals_to_RAI(
        df, df_target_column=None, image_X: dict = {},
        image_y: dict = {}, clear_nans=True, extra_symbols="?", normalize=None,
        max_categorical_threshold=None, text_columns=[]
):
    if max_categorical_threshold:
        for col in df:
            if len(df[col].unique()) > max_categorical_threshold:
                if col not in text_columns:
                    df[col] = pd.Categorical(df[col])
    if normalize is not None:
        if normalize == "Scalar":
            num_d = df.select_dtypes(exclude=['object', 'category'])
            df[num_d.columns] = StandardScaler().fit_transform(num_d)

    output_feature = []
    if df_target_column:
        y = df.pop(df_target_column)
        categorical = str(y.dtypes) in ["object", "category"]
        f = None
        if y.name in text_columns:
            f = Feature(y.name, "text", y.name)
            y = y.tolist()
        elif categorical:
            fact = y.factorize(sort=True)
            f = Feature(
                y.name, "numeric", y.name, categorical=True,
                values={i: v for i, v in enumerate(fact[1])}
            )
            y, _ = y.factorize(sort=True)
        else:
            f = Feature(y.name, "numeric", y.name)
            y = y.tolist()
        output_feature.append(f)
    else:
        y = None
    if image_y is not None:
        for key in image_y:
            f = Feature(key, "image", key)
            image_y[key] = image_y[key].to_numpy()
            y = image_y[key]
    features = []

    for c in df:
        if c in text_columns:
            f = Feature(c, "text", c)
        elif str(df.dtypes[c]) in ["object", "category"]:
            fact = df[c].factorize(sort=True)
            df[c] = fact[0]
            f = Feature(c, "numeric", c, categorical=True,
                        values={i: v for i, v in enumerate(fact[1])})
        elif "float" in str(df.dtypes[c]):
            f = Feature(c, "numeric", c)
        features.append(f)
    if image_X is not None:
        for key in image_X:
            f = Feature(key, "image", key)
            features.append(f)
    return MetaDatabase(features), df.to_numpy(), y, output_feature


# ===== METRIC RELATED UTIL FUNCTIONS =====

# Creates a dictionary where values are mapped using the mapping to specific features
def map_to_feature_dict(values, features, mapping):
    result = {}
    for feature in features:
        result[feature.name] = None
    for i in range(len(values)):
        result[features[mapping[i]].name] = values[i]
    return result


# Creates an array of length features, where features are mapped to specific positions using a map.
# Assumes that values are mapped and are of the same length of mapping.
def map_to_feature_array(values, features, mapping):
    result = [None] * len(features)
    for i in range(len(values)):
        result[mapping[i]] = values[i]
    return result


# Runs a function on each column of data, and returns an array with the value per each column.
def calculate_per_feature(function, X, *args, **kwargs):
    result = []
    for i in range(np.shape(X)[1]):
        result.append(function(X[:, i], *args, **kwargs))
    return result


# Accepts a function, a mapping to features, all features, masked data and arguments.
# If to_array is true, a feature array will be returned, otherwise a feature dict will be returned.
def calculate_per_mapped_features(function, mapping, features, X, *args, to_array=True, **kwargs):
    result = []
    for i in range(np.shape(X)[1]):
        result.append(function(X[:, i], *args, **kwargs))
    if to_array:
        return map_to_feature_array(result, features, mapping)
    else:
        return map_to_feature_dict(result, features, mapping)


# Converts a feature array to a feature dictionary, with one value per feature.
def convert_to_feature_dict(values, features):
    result = {}
    for i, feature in enumerate(features):
        if isinstance(feature, Feature):
            feature = feature.name
        result[feature] = values[i]
    return result


# Converts values to a dictionary with one value per each value of a feature.
def convert_to_feature_value_dict(values, feature):
    result = {}
    for i in range(len(values)):
        result[feature.values[i]] = values[i]
    return result


# Converts float32 values to float64 values for single values, lists/np.arrays and dictionaries.
def convert_float32_to_float64(values):
    if isinstance(values, np.float32):
        return np.float64(values)
    if isinstance(values, list) or isinstance(values, np.ndarray):
        for i in range(len(values)):
            if isinstance(values[i], np.float32):
                values[i] = np.float64(values[i])
        return values
    if isinstance(values, dict):
        for key in values:
            values[key] = convert_float32_to_float64(values[key])
        return values
    return values


class MonitoringTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
