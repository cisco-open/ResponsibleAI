from RAI.metrics.metric_group import MetricGroup
import scipy.stats
from RAI.utils.utils import calculate_per_all_features


# Move config to external .json? 
_config = {
    "name": "correlation_stats_regression",
    "display_name" : "Correlation Stats for Regression Metrics",
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
            if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
                args = self.ai_system.metric_manager.user_config["stats"]["args"]
            data = data_dict["data"]
            map = self.ai_system.meta_database.scalar_map
            features = self.ai_system.meta_database.features

            self.metrics["pearson-correlation"].value = calculate_per_all_features(scipy.stats.pearsonr, data.scalar, data.y, map, features)
            self.metrics["spearman-correlation"].value = calculate_per_all_features(scipy.stats.spearmanr, data.scalar, data.y, map, features)
