from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import scipy.stats

# Move config to external .json? 
_config = {
    "name": "correlation_stats_regression_slow",
    "compatibility": {"type_restriction": "regression", "output_restriction": None},
    "dependency_list": [],
    "tags": ["stats"],
    "complexity_class": "nLog(n)",
    "metrics": {
        "siegel-slopes": {
            "display_name": "Siegel Estimator",
            "type": "vector",
            "has_range": False,
            "range": None,
            "explanation": "",
        },
        "theil-slopes": {
            "display_name": "Theil-Sen Estimator",
            "type": "vector",
            "has_range": False,
            "range": None,
            "explanation": "",
        },
    }
}

# Type (Regression, Classification, Data | probability, numeric)


class CorrelationStatRegressionSlow(MetricGroup, config=_config):
    compatibility = {"type_restriction": None, "output_restriction": None}

    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.user_config is not None and "stats" in self.ai_system.user_config and "args" in self.ai_system.user_config["stats"]:
                args = self.ai_system.user_config["stats"]["args"]
            data = data_dict["data"]
            self.metrics["siegel-slopes"].value = _calculate_per_feature(scipy.stats.siegelslopes, data.X, data.y)
            self.metrics["theil-slopes"].value = _calculate_per_feature(scipy.stats.theilslopes, data.X, data.y)


def _calculate_per_feature(function, X, y):
    result = []
    for i in range(np.shape(X)[1]):
        result.append(function(X[:, i], y))
    return result
