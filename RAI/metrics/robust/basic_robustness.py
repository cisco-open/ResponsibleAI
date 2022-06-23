from RAI.metrics.metric_group import MetricGroup
import numpy as np
import os


class BasicRobustMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
                args = self.ai_system.metric_manager.user_config["stats"]["args"]
            scalar_data = data_dict["data"].scalar

            mean_v = np.mean(scalar_data, **args.get("mean", {}), axis=0, keepdims=True)
            std_v = np.std(scalar_data, **args.get("covariance", {}), axis=0, keepdims= True )
            max_v = np.max(scalar_data, axis=0, keepdims=True)
            min_v = np.min(scalar_data, axis=0, keepdims=True)

            self.metrics["normalized_feature_std"].value = bool(np.all(np.isclose(std_v, np.ones_like(std_v))) and \
                                                    np.all(np.isclose(mean_v, np.ones_like(mean_v))))

            self.metrics["normalized_feature_01"].value = bool(np.all(np.isclose(max_v, np.ones_like(max_v))) and \
                                                    np.all(np.isclose(min_v, np.zeros_like(min_v))))
