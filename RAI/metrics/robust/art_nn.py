from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import pandas as pd
import numpy as np
from RAI.metrics.AIF360.datasets import BinaryLabelDataset
from RAI.metrics.AIF360.metrics import BinaryLabelDatasetMetric

from art.estimators.classification import SklearnClassifier
from art.metrics import empirical_robustness, clever_t, clever_u, clever, loss_sensitivity, wasserstein_distance

R_L1 = 40
R_L2 = 2
R_LI = 0.1

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name": "adversarial_classification_art",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "art",
    "dependency_list": [],
    "tags": ["robustness", "Adversarial"],
    "complexity_class": "linear",
    "metrics": {
        "clever-t-l1": {
            "display_name": "Targeted L1 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-t-l2": {
            "display_name": "Targeted L2 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-t-li": {
            "display_name": "Targeted Li CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-u-l1": {
            "display_name": "Targeted L1 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-u-l2": {
            "display_name": "Targeted L2 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-u-li": {
            "display_name": "Targeted Li CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "loss-sensitivity": {
            "display_name": "Loss Sensitivity",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the number of instances classified"
        },
        "wasserstein-distance": {
            "display_name": "Wassterstein Distance",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the number of positive instances predicted."
        },
        "empirical-robustness": {
            "display_name": "Empirical Robustness",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
    }
}


class ArtAdversarialRobustnessGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        return compatible

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]

            classifier = SklearnClassifier(model=self.ai_system.task.model.agent)
            # CLEVER PARAMS: classifier, input sample, target class, estimate repetitions, random examples to sample per batch, radius of max pertubation, param norm, Weibull distribution init, pool_factor

            example_num = -1
            for i in range(len(preds)):
                if preds[i] != data.y[i]:
                    example_num = i
                    break

        # self.metrics['wasserstein-distance'].value = wasserstein_distance(data.X, data.y)
        self.metrics['loss-sensitivity'].value = loss_sensitivity(classifier, data.X, data.y)
        params = {"eps_step": 1.0, "eps": 1.0}
        self.metrics['empirical-robustness'].value = empirical_robustness(classifier, data.X)

'''
            if example_num != -1:
                self.metrics['clever-t-l1'].value = clever_t(classifier, data.X[example_num], data.y[example_num], 10, 5, R_L1, norm=1, pool_factor=3)
                self.metrics['clever-t-l2'].value = clever_t(classifier, data.X[example_num], data.y[example_num], 10, 5, R_L2, norm=2, pool_factor=3)
                self.metrics['clever-t-li'].value = clever_t(classifier, data.X[example_num], data.y[example_num], 10, 5, R_LI, norm=np.inf, pool_factor=3)
                self.metrics['clever-u-l1'].value = clever_u(classifier, data.X[example_num], data.y[example_num], 10, 5, R_L1, norm=1, pool_factor=3, verbose=False)
                self.metrics['clever-u-l2'].value = clever_u(classifier, data.X[example_num], data.y[example_num], 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
                self.metrics['clever-u-li'].value = clever_u(classifier, data.X[example_num], data.y[example_num], 10, 5, R_LI, norm=np.inf, pool_factor=3, verbose=False)
'''

