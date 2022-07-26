from RAI.metrics.metric_group import MetricGroup
from RAI.utils import map_to_feature_array, map_to_feature_dict, convert_float32_to_float64
import scipy.stats
import os
import numpy as np


class StatMomentGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        args = {}
        if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
            args = self.ai_system.metric_manager.user_config["stats"]["args"]
        data = data_dict["data"]
        scalar_data = data.scalar
        scalar_map = self.ai_system.meta_database.scalar_map
        features = self.ai_system.meta_database.features

        print("Data: ", scalar_data)

        self.metrics["moment_1"].value = map_to_feature_dict(scipy.stats.moment(scalar_data, 1), features, scalar_map)
        self.metrics["moment_2"].value = map_to_feature_dict(scipy.stats.moment(scalar_data, 2), features, scalar_map)
        self.metrics["moment_3"].value = map_to_feature_dict(scipy.stats.moment(scalar_data, 3), features, scalar_map)
        self.metrics["moment_1"].value = convert_float32_to_float64(self.metrics["moment_1"].value)
        self.metrics["moment_2"].value = convert_float32_to_float64(self.metrics["moment_2"].value)
        self.metrics["moment_3"].value = convert_float32_to_float64(self.metrics["moment_3"].value)
