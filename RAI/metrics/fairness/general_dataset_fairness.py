from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import pandas as pd
from RAI.metrics.AIF360.datasets import BinaryLabelDataset
from RAI.metrics.AIF360.metrics import BinaryLabelDatasetMetric

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name": "dataset_fairness",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness", "Data Fairness"],
    "complexity_class": "linear",
    "metrics": {
        "base-rate": {
            "display_name": "Base Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "consistency": {
            "display_name": "Data Consistency",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Shows how similar labels are for similar instances"
        },
        "num-instances": {
            "display_name": "Num Instances",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-negatives": {
            "display_name": "Num Negatives",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-positives": {
            "display_name": "Num Positives",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
    }
}


class GeneralDatasetFairnessGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible \
                     and "fairness" in ai_system.user_config \
                     and "protected_attributes" in ai_system.user_config["fairness"]
        return compatible

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            priv_group = []
            prot_attr = []
            if self.ai_system.user_config is not None and "fairness" in self.ai_system.user_config and "priv_group" in self.ai_system.user_config["fairness"]:
                prot_attr = self.ai_system.user_config["fairness"]["protected_attributes"]

            bin_dataset = get_bin_dataset(self, data, prot_attr)
            self.metrics['base-rate'].value = bin_dataset.base_rate()
            self.metrics['consistency'].value = bin_dataset.consistency()[0]
            self.metrics['num-instances'].value = bin_dataset.num_instances()
            self.metrics['num-negatives'].value = bin_dataset.num_negatives()
            self.metrics['num-positives'].value = bin_dataset.num_positives()


def get_bin_dataset(metric_group, data, prot_attr):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df = pd.DataFrame(data.X, columns=names)
    df['y'] = data.y
    binDataset = BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=prot_attr)
    return BinaryLabelDatasetMetric(binDataset)
