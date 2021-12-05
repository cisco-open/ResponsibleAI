import json
import os.path

from RAI import certificates

__all__ = ['Certificate']




# class Certificate(object):
 
class Certificate(object):
    def __init__(self) -> None:
        super().__init__()
        
        
        self.cert_spec ={}
      
        # self.pre
    def load_from_json(self, json_file):
        
        f = open( json_file, "r")
        self.cert_spec = json.load(f)
        

    def evaluate( self, metrics, certs):
        cond = self.cert_spec["condition"]
        return self._do_eval(cond, metrics, certificates)

    def _do_eval(self, cond, metrics, certs):
        op  = cond["op"] 
        terms = cond["terms"]

        values = []
        for t in terms:
            if isinstance(t,dict):
                values.append( self._do_eval(t, metrics, certs))
            else:
                values.append(self._eval_atomic(t, metrics, certs) )

        if op == "and" :
            return all(values)
        if op == "or" :
            return any(values)
        raise Exception("unknown data type : {}".format(cond))
    
    def _eval_atomic(self, t, metrics, certs):
        var, op, value = t
        
        if value == "True" :
            value = 1

        if value == "False" :
            value = 0


        value = float(value)
        v = self._get_value( var, metrics, certs)
        if op == "==" or op == "=" :
            return v==value
        if op == ">":
            return v > value
        if op == ">=":
            return v >= value
        if op == "<":
            return v > value
        if op == "<=":
            return v <= value

    def _get_value(self, var, metrics, certs):
        if var[0]=="&":
            return metrics[var[1:]]
        if var[0]=="@":
            return certs[var[1:]]

