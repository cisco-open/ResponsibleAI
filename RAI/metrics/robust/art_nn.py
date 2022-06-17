from RAI.metrics.metric_group import MetricGroup
import numpy as np
import torch
import math
from art.estimators.classification import PyTorchClassifier
from art.metrics import empirical_robustness, clever_t, clever_u, clever, loss_sensitivity, wasserstein_distance
from RAI.utils import compare_runtimes

R_L1 = 40
R_L2 = 2
R_LI = 0.1

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name": "adversarial_classification_art",
    "display_name" : "Adverserial Classification (ART) Metrics",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "art",
    "dependency_list": [],
    "tags": ["robustness", "Adversarial"],
    "complexity_class": "polynomial",
    "metrics": {
        "clever-t-l1": {
            "display_name": "Targeted L1 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-t-l2": {
            "display_name": "Targeted L2 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-t-li": {
            "display_name": "Targeted Li CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-u-l1": {
            "display_name": "Untargeted L1 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-u-l2": {
            "display_name": "Untargeted L2 CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
        "clever-u-li": {
            "display_name": "Untargeted Li CLEVER",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        }
    }
}


# Other metrics to consider adding
'''
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
            "explanation": "Measures distribution samples between two inputs."
        },
        "empirical-robustness": {
            "display_name": "Empirical Robustness",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Calculates the rate at which at which a groups with a protected attribute recieve a positive outcome."
        },
'''


class ArtAdversarialRobustnessGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        self.MAX_COMPUTES = 10

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible and 'torch.nn' in str(ai_system.task.model.agent.__class__.__bases__) \
                     and compare_runtimes(ai_system.user_config.get("time_complexity"), _config["complexity_class"])
        return compatible

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]

            classifier = PyTorchClassifier(model=self.ai_system.task.model.agent,
                                                                                 loss=self.ai_system.task.model.loss_function,
                                                                                 optimizer=self.ai_system.task.model.optimizer,
                                                                                 input_shape=[1, 30], nb_classes=2)

            # CLEVER PARAMS: classifier, input sample, target class, estimate repetitions, random examples to sample per batch, radius of max pertubation, param norm, Weibull distribution init, pool_factor

            '''
            X_t = torch.from_numpy(data.X).to(torch.float32).to("cpu")
            n_values = np.max(data.y) + 1
            y_one_hot = np.eye(n_values)[data.y]
            y_t = torch.from_numpy(y_one_hot).to(torch.long).to("cpu")
            '''

            preds = preds.to("cpu")
            example_nums = []
            for i in range(len(preds)):
                if preds[i] != data.y[i]:
                    example_nums.append(i)

            #self.metrics['wasserstein-distance'].value = wasserstein_distance(preds, data.y)
            #self.metrics['loss-sensitivity'].value = loss_sensitivity(classifier, X_t, y_t)
            #params = {"eps_step": 1.0, "eps": 1.0}
            #self.metrics['empirical-robustness'].value = empirical_robustness(classifier, X_t, attack_name="fgsm")

            if len(example_nums) >= 1:
                self.metrics['clever-t-l1'].value = 0
                self.metrics['clever-t-l2'].value = 0
                self.metrics['clever-t-li'].value = 0
                self.metrics['clever-u-l1'].value = 0
                self.metrics['clever-u-l2'].value = 0
                self.metrics['clever-u-li'].value = 0
                to_compute = self.get_selection(example_nums)
                for example_num in to_compute:
                    self.metrics['clever-t-l1'].value += clever_t(classifier, np.float32(data.X[example_num]), data.y[example_num], 10, 5, R_L1, norm=1, pool_factor=3)
                    self.metrics['clever-t-l2'].value += clever_t(classifier, np.float32(data.X[example_num]), data.y[example_num], 10, 5, R_L2, norm=2, pool_factor=3)
                    self.metrics['clever-t-li'].value += clever_t(classifier, np.float32(data.X[example_num]), data.y[example_num], 10, 5, R_LI, norm=np.inf, pool_factor=3)
                    self.metrics['clever-u-l1'].value += clever_u(classifier, np.float32(data.X[example_num]), 10, 5, R_L1, norm=1, pool_factor=3, verbose=False)
                    self.metrics['clever-u-l2'].value += clever_u(classifier, np.float32(data.X[example_num]), 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
                    self.metrics['clever-u-li'].value += clever_u(classifier, np.float32(data.X[example_num]), 10, 5, R_LI, norm=np.inf, pool_factor=3, verbose=False)

                self.metrics['clever-t-l1'].value /= len(to_compute)
                self.metrics['clever-t-l2'].value /= len(to_compute)
                self.metrics['clever-t-li'].value /= len(to_compute)
                self.metrics['clever-u-l1'].value /= len(to_compute)
                self.metrics['clever-u-l2'].value /= len(to_compute)
                self.metrics['clever-u-li'].value /= len(to_compute)

    def get_selection(self, list):
        result = []
        max_items = min(len(list), self.MAX_COMPUTES)
        grab_each = math.floor(len(list)/max_items)
        for i in range(0, len(list), grab_each):
            result.append(list[i])
            if len(result) >= max_items:
                break
        return result
