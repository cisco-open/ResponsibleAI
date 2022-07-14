import os

import numpy as np

from RAI.metrics.metric_group import MetricGroup


class BasicExplainablityGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in \
                    self.ai_system.metric_manager.user_config["stats"]:
                args = self.ai_system.metric_manager.user_config["stats"]["args"]

            scalar_data = data_dict["data"].scalar
            mean_v = np.mean(scalar_data, **args.get("mean", {}), axis=0, keepdims=True)
            std_v = np.std(scalar_data, **args.get("covariance", {}), axis=0, keepdims=True)
            max_v = np.max(scalar_data, axis=0, keepdims=True)
            min_v = np.min(scalar_data, axis=0, keepdims=True)

            self.metrics["explainable_model"].value = True

# TODO: This class is a placeholder for Explainability functions. Clarify/remove this class
