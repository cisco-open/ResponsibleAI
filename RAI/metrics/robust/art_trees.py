import warnings
from RAI.metrics.metric_group import MetricGroup
import numpy as np
from art.estimators.classification import SklearnClassifier
from art.metrics.verification_decisions_trees import RobustnessVerificationTreeModelsCliqueMethod
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

R_L1 = 40
R_L2 = 2
R_LI = 0.1


class ArtAdversarialRobustnessTreeGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def is_compatible(ai_system):
        compatible = super().is_compatible(ai_system)
        return compatible and ai_system.model.agent.__class__.__module__.split(".")[0] == "sklearn"

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]

            classifier = SklearnClassifier(model=self.ai_system.model.agent)
            rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)

            if data.y.ndim == 1:
                y = np.stack([data.y == 0, data.y == 1], 1)
            else:
                y = data.y
            self.metrics['adversarial_tree_verification_bound'].value, \
            self.metrics['adversarial_tree_verification_error'].value = \
                rt.verify(data.X, y, eps_init=0.3, nb_search_steps=2, max_clique=2, max_level=2)
