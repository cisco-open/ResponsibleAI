from RAI.metrics.ai360_helper import get_binary_dataset
from RAI.metrics.metric_group import MetricGroup
import os


class GeneralDatasetFairnessGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    @classmethod
    def is_compatible(cls, ai_system):
        compatible = super().is_compatible(ai_system)
        return compatible \
            and "fairness" in ai_system.metric_manager.user_config \
            and "protected_attributes" in ai_system.metric_manager.user_config["fairness"] \
            and len(ai_system.metric_manager.user_config["fairness"]["protected_attributes"]) > 0

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        data = data_dict["data"]
        prot_attr = []
        if self.ai_system.metric_manager.user_config is not None and "fairness" in self.ai_system.metric_manager.user_config and "priv_group" in \
                self.ai_system.metric_manager.user_config["fairness"]:
            prot_attr = self.ai_system.metric_manager.user_config["fairness"]["protected_attributes"]

        bin_dataset = get_binary_dataset(self, data, prot_attr)
        self.metrics['base_rate'].value = bin_dataset.base_rate()
        self.metrics['num_instances'].value = bin_dataset.num_instances()
        self.metrics['num_negatives'].value = bin_dataset.num_negatives()
        self.metrics['num_positives'].value = bin_dataset.num_positives()
