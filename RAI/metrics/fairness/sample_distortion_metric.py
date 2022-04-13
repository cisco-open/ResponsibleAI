from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import pandas as pd
from RAI.metrics.AIF360.datasets import BinaryLabelDataset
from RAI.metrics.AIF360.metrics import BinaryLabelDatasetMetric
from RAI.utils import compare_runtimes

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}


# IMPROVE CONFIG ONCE METRICS ARE SET UP.
_config = {
    "name": "sample_distortion_fairness",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness", "General Fairness"],
    "complexity_class": "linear",
    "metrics": {
        "average": {
            "display_name": "Average",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
    }
}


class SampleDistortionFairnessGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible \
                     and "fairness" in ai_system.metric_manager.user_config \
                     and "protected_attributes" in ai_system.metric_manager.user_config["fairness"] \
                     and  len(ai_system.metric_manager.user_config["fairness"]["protected_attributes"])>0 \
                     and "positive_label" in ai_system.metric_manager.user_config["fairness"] \
                     and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"), _config["complexity_class"])
        return compatible

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            prot_attr = []
            if self.ai_system.metric_manager.user_config is not None and "fairness" in self.ai_system.metric_manager.user_config and "priv_group" in \
                    self.ai_system.metric_manager.user_config["fairness"]:
                prot_attr = self.ai_system.metric_manager.user_config["fairness"]["protected_attributes"]
                pos_label = self.ai_system.metric_manager.user_config["fairness"]["positive_label"]

            # bin_dataset = get_bin_dataset(self, data, priv_group)

            self.metrics['average'].value = 0


def get_bin_dataset(metric_group, data, priv_group):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df = pd.DataFrame(data.X, columns=names)
    df['y'] = data.y
    binDataset = BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=['race'])
    return BinaryLabelDatasetMetric(binDataset)
