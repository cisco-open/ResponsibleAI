from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np


# Move config to external .json? 
_config = {
    "src": "stats",
    "category": "bias",
    "dependency_list": [],
    "tags": ["stats"],
    "complexity_class": "linear",
    "metrics": {
        "mean": {
            "type": "other",
            "tags": [],
            "has_range": False,
            "range": None,
            "explanation": "Mean is the expected value of data.",
        },
        "covariance": {
            "type": "other",
            "tags": [],
            "has_range": False,
            "range": None,
            "explanation": "A Covariance Matrix shows the directional relationship between different data members.",
        },

        "num-Nan-rows": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Num Nan Rows indicates the number of NaN rows found in the data.",
        },
        "percent-Nan-rows": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Percent Nan Rows indicates the percentage of rows that are NaN in the data.",
        },
    }
}

# Type (Regression, Classification, Data | probability, numeric)


class StatMetricGroup(MetricGroup, name="stat"):
    compatibility = {"type_restriction": None, "output_restriction": None}

    def __init__(self, ai_system, config=_config) -> None:
        super().__init__(ai_system, config)
        self.ai_system = ai_system
        self.compatibility = {"type_restriction": None, "output_restriction": None}

    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if "stats" in self.ai_system.user_config and "args" in self.ai_system.user_config["stats"]:
                args = self.ai_system.user_config["stats"]["args"]

            data = data_dict["data"]
            self.metrics["mean"].value = np.mean(data.X, **args.get("mean", {}))
            self.metrics["covariance"].value = np.cov(data.X, **args.get("covariance", {}))
            self.metrics["num-Nan-rows"].value = np.count_nonzero(np.isnan(data.X).any(axis=1))
            self.metrics["percent-Nan-rows"].value = self.metrics["num-Nan-rows"].value/np.shape(np.asarray(data.X))[0]

