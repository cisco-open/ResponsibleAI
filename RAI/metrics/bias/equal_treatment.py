from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import sklearn

__all__ = ['compatibility']

compatibility = {"type_restriction": "classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "src": "equal_treatment",
    "category": "bias",
    "dependency_list": [],
    "tags": ["Equal Treatment"],
    "complexity_class": "linear",
    "metrics": {
        "disparate_impact_ratio": {
            "display_name": "Disparate Impact",
            "type": "numeric",
            "tags": ["Equal Treatment"],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Disparate Impact describes how preferential the treatment is for privileged groups when compared to unprivileged groups."
        }
    }
}


class EqualTreatmentMetricGroup(MetricGroup, name="equal_treatment"):
    def __init__(self, ai_system, config=_config) -> None:
        super().__init__(ai_system, config)
        self.ai_system = ai_system
        self.config = config
        self.compatibility = {"type_restriction": "classification", "output_restriction": "choice"}

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            args = {}
            if self.ai_system.user_config is not None and "bias" in self.ai_system.user_config and "args" in self.ai_system.user_config["bias"]:
                args = self.ai_system.user_config["bias"]["args"]
            self.metrics["disparate_impact_ratio"].value = _disparate_impact_ratio()


def _disparate_impact_ratio():
    return 0

