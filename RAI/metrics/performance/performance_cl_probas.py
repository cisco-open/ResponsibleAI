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
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(data.y, probs[:, 1])
        self.metrics["auc"].value = sklearn.metrics.auc(fpr, tpr)
