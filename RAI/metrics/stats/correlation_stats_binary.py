from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import scipy.stats
from RAI.utils.utils import calculate_per_all_features


# Are these metrics meaningful?

_config = {
    "name": "correlation_stats_binary",
    "display_name" : "Binary Classification Metrics",
    "compatibility": {"type_restriction": "binary_classification", "output_restriction": None},
    "dependency_list": [],
    "tags": ["stats", "Binary Correlation"],
    "complexity_class": "linear",
    "metrics": {
        "point-biserial-r": {
            "display_name": "Point Biserial Coefficient",
            "type": "vector",
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Indicates the relationship between a binary variable and a continuous variable.",
        },
    }
}

# Type (Regression, Classification, Data | probability, numeric)


class BinaryCorrelationStats(MetricGroup, config=_config):
    compatibility = {"type_restriction": "binary_classification", "output_restriction": None}

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
            map = self.ai_system.meta_database.categorical_map
            features = self.ai_system.meta_database.features

            self.metrics["point-biserial-r"].value = calculate_per_all_features(scipy.stats.pointbiserialr, data.categorical, data.y, map, features)
