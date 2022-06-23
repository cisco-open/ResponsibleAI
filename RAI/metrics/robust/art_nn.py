from RAI.metrics.metric_group import MetricGroup
import numpy as np
import math
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_t, clever_u
from RAI.utils import compare_runtimes
import os


R_L1 = 40
R_L2 = 2
R_LI = 0.1


class ArtAdversarialRobustnessGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        self.MAX_COMPUTES = 10

    def is_compatible(ai_system):
        compatible = super().is_compatible(ai_system)
        return compatible and 'torch.nn' in str(ai_system.model.agent.__class__.__bases__)

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]

            classifier = PyTorchClassifier(model=self.ai_system.model.agent, loss=self.ai_system.model.loss_function,
                                           optimizer=self.ai_system.model.optimizer, input_shape=[1, 30], nb_classes=2)
            # TODO: Remove limitation on input shape

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
