import numpy as np
import math
__all__ = [ 'jsonify']


def jsonify(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) in (np.bool,'_bool'):
            return bool(v)
        if type(v) is float and math.isinf(v):  # CURRENTLY REPLACING INF VALUES WITH NULL
            return None

        return v

