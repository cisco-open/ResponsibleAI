import numpy as np
__all__ = [ 'jsonify']


def jsonify(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) in (np.bool,'_bool'):
            return bool(v)
        
        return v