import numpy as np
import math
__all__ = [ 'jsonify', 'compare_runtimes']


def jsonify(v):
        if type(v) is np.ma.MaskedArray:
            return np.ma.getdata(v).tolist()
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) in (np.bool, '_bool', 'bool_') or v.__class__.__name__ == "bool_":
            return bool(v)
        if (isinstance(v, int) or isinstance(v, float)) and (math.isinf(v) or math.isnan(v)):  # CURRENTLY REPLACING INF VALUES WITH NULL
            return None
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
