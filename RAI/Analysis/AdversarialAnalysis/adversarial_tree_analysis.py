from RAI.Analysis import Analysis
from RAI.AISystem import AISystem
from art.metrics.verification_decisions_trees import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification.scikitlearn import SklearnClassifier
import os
import numpy as np


class AdversarialTreeAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.distortion_size = 0.3

    @classmethod
    def is_compatible(cls, ai_system: AISystem, dataset: str):
        compatible = super().is_compatible(ai_system, dataset)
        return compatible and str(ai_system.model.agent.__class__).startswith("<class 'sklearn.ensemble.")

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        data = self.ai_system.get_data(self.dataset)
        classifier = SklearnClassifier(model=self.ai_system.model.agent)
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        if data.y.ndim == 1:
            y = np.stack([data.y == 0, data.y == 1], 1)
        else:
            y = data.y

        # Note: This runs slow, to speed it up we can take portion of test set size
        result['adversarial_tree_verification_bound'], result['adversarial_tree_verification_error'] = \
            rt.verify(data.X, y, eps_init=self.distortion_size, nb_search_steps=10, max_clique=2, max_level=2)
        return result

    def to_string(self):
        result = "\n==== Decision Tree Adversarial Analysis ====\n"
        result += "This test uses the Clique Method Robustness Verification method.\n" \
                  "The Adversarial Tree Verification Lower Bound describes the lower bound of " \
                  "minimum L-infinity adversarial distortion averaged over all test examples.\n"
        result += "Adversarial Tree Verification Lower Bound: " + str(self.result['adversarial_tree_verification_bound'])\
                  + '\n'
        result += "\nAdversarial Tree Verified Error is the upper bound of error under any attacks.\n" \
                  "Verified Error guarantees that within a L-infinity distortion norm of " + str(self.distortion_size)+\
                  ", that no attacks can achieve over X% error on test sets.\n"
        result += "Adversarial Tree Verified Error: " + str(self.result['adversarial_tree_verification_error']) + "\n"
        return result

    def to_html(self):
        pass
