from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np


# Move config to external .json? 
_config = {
    "src": "stats",
    "dependency_list": [],
    "tags": ["stats"],
    "complexity_class": "linear",
    "metrics": {
        "mean": {
            "type": "other",
            "tags": [],
            "has_range": False,

        },
        "covariance": {
            "type": "other",
            "tags": [],
            "has_range": False,
        },

        "num-Nan-rows": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
        },
        "percent-Nan-rows": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
    }
}


_data_dependent = {
    "disparate-impact": {
        "type": "numeric",
        "tags": [],
        "has_range": True,
        "range": [0, 1]
    }
}


class StatMetricGroup(MetricGroup, name="stat"):
    def __init__(self, ai_system, config=_config) -> None:
        new_config = _generate_config(config, ai_system.user_config)
        super().__init__(ai_system, new_config)

    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            data = data_dict["data"]
            self.metrics["mean"].value = np.mean(data.X)
            self.metrics["covariance"].value = np.cov(data.X)
            self.metrics["num-Nan-rows"].value = np.count_nonzero(np.isnan(data.X).any(axis=1))
            self.metrics["percent-Nan-rows"].value = self.metrics["num-Nan-rows"].value/np.shape(np.asarray(data.X))[0]

def _generate_config(config, user_config):
    if config["src"] != "stats":
        return config
    res = config
    return res

