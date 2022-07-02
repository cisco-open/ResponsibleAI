import math
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from RAI.dataset import Feature, MetaDatabase

__all__ = ['jsonify', 'compare_runtimes', 'df_to_meta_database', 'df_to_RAI', 'Reweighing',
           'calculate_per_mapped_features',
           'convert_to_feature_value_dict', 'convert_to_feature_dict', 'map_to_feature_array', 'map_to_feature_dict']


def Reweighing():
    pass


def isPrimitive(obj):
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
    if isPrimitive(v):
        return v
    # if isPrimitive(v):
    return pickle.dumps(v).decode('ISO-8859-1')
    return v


def clean_list(v):
    for i in range(len(v)):
        v[i] = jsonify(v[i])
    return v


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


def df_to_meta_database(df, categorical_values=None, protected_attribute_names=None, privileged_info=None,
                        positive_label=None):
    features = []
    fairness_config = {}
    for col in df.columns:
        categorical = categorical_values is not None and col in categorical_values
        features.append(Feature(col, "float32", col, categorical=categorical, values=categorical_values.get(col, None)))
    if protected_attribute_names != None:
        fairness_config["protected_attributes"] = protected_attribute_names
    if privileged_info != None:
        fairness_config["priv_group"] = privileged_info
    if positive_label != None:
        fairness_config["positive_label"] = positive_label
    meta = MetaDatabase(features)
    return meta, fairness_config


def df_remove_nans(df, extra_symbols):
    for i in df:
        df[i].replace('nan', np.nan, inplace=True)
        for s in extra_symbols:
            df[i].replace(s, np.nan, inplace=True)
    df.dropna(inplace=True)


def df_to_RAI(df, test_tf=None, target_column=None, clear_nans=True, extra_symbols="?", normalize="Scalar",
              max_categorical_threshold=None):
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
    features = []
    cat_columns = []
    if target_column:
        y = df.pop(target_column)
        if y.dtype in ("object", "category"):
            y = y.factorize(sort=True)[0]
    else:
        y = None

    features = []

    for c in df:
        if str(df.dtypes[c]) in ["object", "category"]:
            fact = df[c].factorize(sort=True)
            df[c] = fact[0]
            f = Feature(c, "integer", c, categorical=True,
                        values={i: v for i, v in enumerate(fact[1])})
        else:
            f = Feature(c, "float", c)
        features.append(f)
    return MetaDatabase(features), df.to_numpy().astype('float32'), y


def map_to_feature_dict(values, features, mapping):
    result = {}
    for feature in features:
        result[feature] = None
    for i in range(len(values)):
        result[features[mapping[i]]] = values[i]
    return result


def map_to_feature_array(values, features, mapping):
    result = [None] * len(features)
    for i in range(len(values)):
        result[mapping[i]] = values[i]
    return result


def calculate_per_feature(function, X, *args, **kwargs):
    result = []
    for i in range(np.shape(X)[1]):
        result.append(function(X[:, i], *args, **kwargs))
    return result


def calculate_per_mapped_features(function, mapping, features, X, *args, to_array=True, **kwargs):
    result = []
    for i in range(np.shape(X)[1]):
        result.append(function(X[:, i], *args, **kwargs))
    if to_array:
        return map_to_feature_array(result, features, mapping)
    else:
        return map_to_feature_dict(result, features, mapping)


def convert_to_feature_dict(values, features):
    result = {}
    for i, feature in enumerate(features):
        result[feature] = values[i]
    return result


def convert_to_feature_value_dict(values, feature):
    result = {}
    for i in range(len(values)):
        result[feature.values[i]] = values[i]
    return result


def convert_float32_to_float64(values):
    for i in range(len(values)):
        if isinstance(values[i], np.float32):
            values[i] = np.float64(values[i])
    return values
