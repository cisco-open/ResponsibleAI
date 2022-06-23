from RAI.metrics.metric_group import MetricGroup
import numpy as np
import sklearn
import os


class AdversarialRobustnessMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
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
            self.metrics["inaccuracy"].value = np.sqrt(1 - sklearn.metrics.accuracy_score(data.y, preds, **args.get("accuracy", {})))
            self.metrics["data-dimensionality"].value = np.sqrt(len(data.X[0]))
            self.metrics["l-inf-model-score"].value = None
            self.metrics["l-2-model-score"].value = None

            # Is Robustness the Cost of Accuracy? A Comprehensive Study on the Robustness of 18 Deep Image Classification Models. - Dong Su
            l_inf_distortions = {"alexnet": 3.5E-2, "vgg": 2.7E-2, "resnet": 1.5E-2, "inception": 1.5E-2, "mobilenet": 5E-3,
                                 "densenet": 7E-3, "nasnet": 1E-2}
            l_2_distortions = {"alexnet": 1.1E-5, "vgg": 8E-6, "resnet": 2E-6, "inception": 3E-6, "mobilenet": 3.5E-6,
                                 "densenet": 4E-6, "nasnet": 2E-6}
            clever_distortions = {"alexnet": 8E-6, "vgg": 6.5E-6, "resnet": 3.7E-6, "inception": 3E-6, "mobilenet": 2E-6,
                                 "densenet": 3E-6, "nasnet": 3.5E-6}

            model_class = self.ai_system.model.model_class
            if model_class in l_inf_distortions:
                self.metrics["l-inf-model-score"].value = l_inf_distortions[model_class]
            if model_class in l_2_distortions:
                self.metrics["l-2-model-score"].value = l_2_distortions[model_class]
            if model_class in clever_distortions:
                self.metrics["clever-model-score"].value = clever_distortions[model_class]

# TODO: Remove these?