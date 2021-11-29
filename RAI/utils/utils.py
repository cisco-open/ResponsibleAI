import numpy as np
import math
__all__ = [ 'jsonify']


def jsonify(v):
        if type(v) is np.ma.MaskedArray:
            return np.ma.getdata(v).tolist()
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) in (np.bool,'_bool'):
            return bool(v)
        if (isinstance(v, int) or isinstance(v, float)) and (math.isinf(v) or math.isnan(v)):  # CURRENTLY REPLACING INF VALUES WITH NULL
            return None


        return v

