from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np


_config = {
    "dependency_list":[],
    "tags":["stats"],
    "complexity_class":"linear",
    "metrics": {
        "mean":{
            "type":"other",
            "tags":[],
            "has_range":False,

        },
        "covariance":{
            "type":"other",
            "tags":[],
            "has_range":False,
        },

        "num-Nan-rows":{
            "type":"numeric",
            "tags":[],
            "has_range":True,
            "range":[0,None],
        },
        "percent-Nan-rows":{
            "type":"numeric",
            "tags":[],
            "has_range":True,
            "range":[0,1],
        },
        
    }

}


class StatMetricGroup(MetricGroup, name = "stat"):
    def __init__(self, ai_system, config = _config) -> None:
        super().__init__(ai_system, config)

    
    def update(self, data):
        pass
    
    def compute(self, data):
        self.metrics["mean"].value = np.mean ( data.X )
        self.metrics["covariance"].value = np.cov( data.X )


        
        