from RAI.metrics.metric_group import MetricGroup
import numpy as np
import sklearn
import os


class AdversarialRobustnessMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        args = {}
        if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
            args = self.ai_system.metric_manager.user_config["stats"]["args"]

        data = data_dict["data"]
        preds = data_dict["predict"]
        self.metrics["inaccuracy"].value = np.sqrt(1 - sklearn.metrics.accuracy_score(data.y, preds, **args.get("accuracy", {})))


# TODO: Add more metrics
