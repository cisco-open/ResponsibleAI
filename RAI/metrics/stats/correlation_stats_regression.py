from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import scipy.stats

# Move config to external .json? 
_config = {
    "name": "correlation_stats_regression",
    "compatibility": {"type_restriction": "regression", "output_restriction": None},
    "dependency_list": [],
    "tags": ["stats", "Regression Correlation"],
    "complexity_class": "linear",
    "metrics": {
        "pearson-correlation": {
            "display_name": "Pearson's Correlation'",
            "type": "vector",
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Indicates if a statistically significance relationship is found between variables.",
        },
        "spearman-correlation": {
            "display_name": "Spearman's Correlation",
            "type": "vector",
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Measures rank correlation. Indicates how well can two variables be described with a monotonic function.",
        },
    }
}

'''  TEMPORARILY REMOVED LINEAR REGRESSION. 
"lin-regress": {
            "display_name": "Linear Regression",
            "type": "vector",
            "has_range": False,
            "range": None,
            "explanation": "",
        },
'''

# Type (Regression, Classification, Data | probability, numeric)


class CorrelationStatRegression(MetricGroup, config=_config):
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

            mask = self.ai_system.meta_database.scalar_mask

            data = data_dict["data"]
            self.metrics["pearson-correlation"].value = _masked_calculate_per_feature(scipy.stats.pearsonr, data.X, data.y, mask)
            self.metrics["spearman-correlation"].value = _masked_calculate_per_feature(scipy.stats.spearmanr, data.X, data.y, mask)
            # self.metrics["lin-regress"].value = _masked_calculate_per_feature(scipy.stats.linregress, data.X, data.y, mask)


def _calculate_per_feature(function, X, y):
    result = []
    for i in range(np.shape(X)[1]):
        result.append(function(X[:, i], y))
    return result

def _masked_calculate_per_feature(function, X, y, mask):
    result = []
    for i in range(np.shape(X)[1]):
        if(mask[i]):
            result.append(function(X[:, i], y))
        else:
            result.append(None)
    return result