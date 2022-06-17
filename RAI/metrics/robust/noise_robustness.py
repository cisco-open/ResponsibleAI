from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import sklearn


# Move config to external .json? 
_config = {
    "name": "noise_robustness",
    "display_name" : "Robustness to Noise Metrics",
    "compatibility": {"type_restriction": None, "output_restriction": None},
    "dependency_list": [],
    "tags": ["robustness", "Noise"],
    "complexity_class": "linear",
    "metrics": {
        "certified-adversarial-noise": {
            "display_name": "Certified Adversarial Robustness with Additive Noise Score",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "https://github.com/Bai-Li/STN-Code",
        },
        "robust-signal-noise-ratio": {
            "display_name": "Robust SNR",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning . https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#snrdistance",
        },
        "decision-curve-upper-bound": {
            "display_name": "Robustness of classifiers: from adversarial to random noise score",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "Robustness of classifiers: from adversarial to random noise. Requires hand coding their math.",
        },
        "decision-curve-lower-bound": {
            "display_name": "Robustness of classifiers: from adversarial to random noise score",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "Robustness of classifiers: from adversarial to random noise. Requires hand coding their math",
        },

        "added-noise": {
            "display_name": "Accuracy After Noise",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "Add Label Noise / Noise proportional to stdev and and compare accuracy?",
        },
    }
}


# Type (Regression, Classification, Data | probability, numeric)
class NoiseRobustnessMetricGroup(MetricGroup, config=_config):
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
            preds = data_dict["predictions"]

# TODO: Clarify or remove?