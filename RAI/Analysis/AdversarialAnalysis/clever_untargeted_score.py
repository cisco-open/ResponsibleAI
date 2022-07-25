import numpy as np
import torch

from RAI.AISystem import AISystem
from RAI.Analysis import Analysis
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_t, clever_u
import os

R_L1 = 40
R_L2 = 2
R_LI = 0.1


class CleverUntargetedScore(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.EXAMPLES_PER_CLASS = 2

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        data = self.ai_system.get_data(self.dataset)
        xData = data.X
        yData = data.y
        output_features = self.ai_system.model.output_features[0].values
        self.output_features = output_features.copy()
        numClasses = len(output_features)
        shape = data.image[0].shape
        classifier = PyTorchClassifier(model=self.ai_system.model.agent, loss=self.ai_system.model.loss_function,
                                       optimizer=self.ai_system.model.optimizer, input_shape=shape, nb_classes=numClasses)
        # This should also be done per class
        result['clever_u_l1'] = {i: [] for i in output_features}
        result['clever_u_l2'] = {i: [] for i in output_features}
        result['clever_u_li'] = {i: [] for i in output_features}

        correct_classifications = self._get_correct_classifications(self.ai_system.model.predict_fun, xData, yData)
        balanced_classifications = self._balance_classifications_per_class(correct_classifications, yData, output_features)

        for target_class in balanced_classifications:
            for example_num in balanced_classifications[target_class]:
                example = data.X[example_num][0]
                result['clever_u_l1'][target_class].append(clever_u(classifier, example, 10, 5, R_L1, norm=1, pool_factor=3, verbose=False))
                result['clever_u_l2'][target_class].append(clever_u(classifier, example, 10, 5, R_L2, norm=2, pool_factor=3, verbose=False))
                result['clever_u_li'][target_class].append(clever_u(classifier, example, 10, 5, R_LI, norm=np.inf, pool_factor=3, verbose=False))
        result['total_images'] = 0
        result['total_classes'] = len(result['clever_u_l1'])
        for val in balanced_classifications:
            result['total_images'] += len(balanced_classifications[val])
        return result

    def _get_correct_classifications(self, predict_fun, xData, yData):
        result = []
        for i, example in enumerate(xData):
            pred = predict_fun(torch.Tensor(example))
            if np.argmax(pred.detach().numpy(), axis=1)[0] == yData[i]:
                result.append(i)
        return result

    def _balance_classifications_per_class(self, classifications, yData, class_values):
        result = {i: [] for i in class_values}
        for classification in classifications:
            if len(result[yData[classification]]) < self.EXAMPLES_PER_CLASS:
                result[yData[classification]].append(classification)
        return result

    def _get_selection(self, misclassifications):
        offset = int(len(misclassifications)/self.MAX_COMPUTES)
        to_compute = [misclassifications[i] for i in range(0, len(misclassifications), offset)]
        return to_compute

    def _result_stats(self, res):
        return "Average Value " + str(sum(res)/len(res)) + ", Minimum Value: " + str(min(res)) + ", Maximum Value: " + str(max(res))

    def to_string(self):
        result = "\n==== CLEVER Untargeted Score Analysis ====\nCLEVER Score is an attack independent robustness metric " \
                 "which can be used to evaluate any neural network.\nCLEVER scores provide a lower bound for adversarial " \
                 "attacks of various norms.\n"
        result += "This evaluation computed the CLEVER score across " + str(self.result['total_images']) + " examples.\n"
        result += "CLEVER untargeted scores describe attacks where the adversary attempts to trick the classifier to pick " \
                  "any incorrect class.\n"
        result += "L1 Perturbations describes the sum of the perturbation size.\n"
        for val in self.result['clever_u_l1']:
            result += "The Untargeted CLEVER L1 score to trick the classifier from correctly picking class " + self.output_features[val] + " is: \n" \
                      + self._result_stats(self.result['clever_u_l1'][val]) + "\n"
        result += "\nL2 Perturbations describes the manhattan distance between the input before and after perturbation.\n"
        for val in self.result['clever_u_l2']:
            result += "The Untargeted CLEVER L2 score to trick the classifier from correctly picking class " + self.output_features[val] + " is: \n" \
                      + self._result_stats(self.result['clever_u_l2'][val]) + "\n"
        result += "\nL-inf Perturbations describes the maximum size of a perturbation.\n"
        for val in self.result['clever_u_li']:
            result += "The Untargeted CLEVER L-inf score to trick the classifier from correctly picking class " + self.output_features[val] + " is: \n" \
                      + self._result_stats(self.result['clever_u_li'][val]) + "\n"
        return result

    def to_html(self):
        pass

