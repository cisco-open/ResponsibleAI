from RAI.metrics.metric_group import MetricGroup
import sklearn
import os


class PerformanceClassificationProbasMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        data = data_dict["data"]
        probs = data_dict["predict_proba"]
        self.metrics["roc_auc"].value = sklearn.metrics.roc_auc_score(data.y, probs[:, 1])

