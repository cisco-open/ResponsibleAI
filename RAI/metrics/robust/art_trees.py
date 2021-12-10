from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import pandas as pd
import numpy as np
import art

from art.estimators.classification import SklearnClassifier
from art.metrics.verification_decisions_trees import RobustnessVerificationTreeModelsCliqueMethod
from RAI.utils import compare_runtimes


R_L1 = 40
R_L2 = 2
R_LI = 0.1

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name": "adversarial_validation_tree",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "art",
    "dependency_list": [],
    "tags": ["robustness", "Adversarial Tree"],
    "complexity_class": "polynomial",
    "metrics": {
        "adversarial-tree-verification-bound": {
            "display_name": "Adversarial Tree Avg Verification Bound",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": 'Lower bound on minimum adversarial distortion averaged over all test samples. Larger is better. Chen, Hongge, et al. "Robustness verification of tree-based models." arXiv preprint arXiv:1906.03849 (2019).'
        },
        "adversarial-tree-verification-error": {
            "display_name": "Adversarial Tree Verified Error",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": 'Upper bound of error for adversarial examples with L_infinity pertubation of specified size. From Chen, Hongge, et al. "Robustness verification of tree-based models." arXiv preprint arXiv:1906.03849 (2019).'
        },
    }
}


class ArtAdversarialRobustnessTreeGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible and ai_system.task.model.agent.__class__.__module__.split(".")[0] == "sklearn" \
                     and compare_runtimes(ai_system.user_config.get("time_complexity"), _config["complexity_class"])

        return compatible

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]

            classifier = SklearnClassifier(model=self.ai_system.task.model.agent)

            rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)

            self.metrics['adversarial-tree-verification-bound'].value, \
            self.metrics['adversarial-tree-verification-error'].value = \
                rt.verify(data.X, np.reshape(data.y, (data.y.size, 1)), eps_init=0.3, nb_search_steps=2, max_clique=2, max_level=2)


