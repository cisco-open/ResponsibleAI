import numpy as np
import math
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
__all__ = [ 'jsonify', 'compare_runtimes', 'df_to_meta_database']


def jsonify(v):
        if type(v) is np.ma.MaskedArray:
            return np.ma.getdata(v).tolist()
        if type(v) is np.ndarray:
            return clean_list(v.tolist())
        if type(v) is list:
            return clean_list(v)
        if type(v) in (np.bool, '_bool', 'bool_') or v.__class__.__name__ == "bool_":
            return bool(v)
        if (isinstance(v, int) or isinstance(v, float)) and (math.isinf(v) or math.isnan(v)):  # CURRENTLY REPLACING INF VALUES WITH NULL
            return None
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


def df_to_meta_database(df, categorical_values=None, protected_attribute_names=None, privileged_info=None, positive_label=None):
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

