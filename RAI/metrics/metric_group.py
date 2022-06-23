from RAI.utils import compare_runtimes
from .metric import Metric
from RAI.metrics.registry import register_class
import numpy as np
import os
import json

__all__ = ['MetricGroup', 'all_complexity_classes', 'all_task_types', 'all_data_types', 'all_output_requirements', 'all_data_requirements']

all_complexity_classes = {"constant",  "linear",  "multi_linear", "polynomial", "exponential"}
all_task_types = {"binary_classification", "classification", "clustering", "regression"}
all_data_types = {"numeric", "image", "text"}
all_output_requirements = {"predict", "predict_proba", "generate_text"}
all_dataset_requirements = {"X", "y", "sensitive_features"}


class MetricGroup(object):    
    name = ""
    config = None

    @classmethod
    def is_compatible(cls, ai_system):
        compatible = cls.config["compatibility"]["task_type"] is None or cls.config["compatibility"]["task_type"] == "" \
                     or cls.config["compatibility"]["task_type"] == ai_system.model.task \
                     or (cls.config["compatibility"]["task_type"] == "classification" and ai_system.model.task == "binary_classification")
        compatible = compatible and (cls.config["compatibility"]["data_type"] is None or cls.config["compatibility"]["data_type"] == [] or\
                     all(item in ai_system.meta_database.data_format for item in cls.config["compatibility"]["data_type"]))
        compatible = compatible and (cls.config["compatibility"]["output_requirements"] is None or \
                     all(item in ai_system.model.output_types for item in cls.config["compatibility"]["output_requirements"]))
        compatible = compatible and (cls.config["compatibility"]["dataset_requirements"] is None or \
                     all(item in ai_system.meta_database.stored_data for item in cls.config["compatibility"]["dataset_requirements"]))
        compatible = compatible and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"), cls.config["complexity_class"])
        return compatible

    def __init_subclass__(cls, class_location=None, **kwargs):
        super().__init_subclass__(**kwargs)
        config_file = class_location[:-2]+"json"
        cls.config = json.load(open(config_file))
        cls.name = cls.config["name"]
        register_class(cls.name, cls)

    def __init__(self, ai_system) -> None:
        self.ai_system = ai_system
        self.persistent_data = {}
        self.dependency_list = []
        self.metrics = {}
        self.tags = []
        self.complexity_class = ""
        self.display_name = self.name
        self.compatiblity = {}
        self.status = "OK"
        self.reset()
        
        if self.load_config(self.config):
            self.status = "OK"
        else:
            self.status = "BAD"

    def reset(self):
        if self.status == "BAD":
            return
        self.persistent_data = {}
        self.value = None
        self.status = "OK"

    def load_config(self, config):
        if "tags" in config:
            self.tags = config["tags"]
        if "dependency_list" in config:
            self.dependency_list = config["dependency_list"]
        if "complexity_class" in config:
            self.complexity_class = config["complexity_class"]
        if "compatibility" in config:
            self.compatiblity = config["compatibility"]
        if "display_name" in config:
            self.display_name = config["display_name"]
        else:
            self.display_name = self.name
        if "metrics" in config:
            self.create_metrics(config["metrics"])
        return True

    def create_metrics(self, metrics_config):
        for metric_name in metrics_config:
            self.metrics[metric_name] = Metric(metric_name, metrics_config[metric_name])
            self.metrics[metric_name].unique_name = self.name + " > " + metric_name
            self.metrics[metric_name].tags = self.tags

    def get_metric_values(self):
        results = {}
        for metric_name in self.metrics:
            if self.metrics[metric_name].type == 'vector':
                results[metric_name + "-single"] = self.metrics[metric_name].value[0]
                val = self.metrics[metric_name].value[1]
                if type(self.metrics[metric_name].value[1]) is np.ndarray:
                    val = val.tolist()
                results[metric_name + "-individual"] = val  # Easily modify to export for each value.
            else:
                results[metric_name] = self.metrics[metric_name].value
        return results

    def export_metric_values(self):
        results={}
        for metric_name in self.metrics:
            if self.metrics[metric_name].type == 'vector':
                results[metric_name + "-single"] = self.metrics[metric_name].value[0]
                val = self.metrics[metric_name].value[1]
                if type(self.metrics[metric_name].value[1]) is np.ndarray:
                    val = val.tolist()
                results[metric_name + "-individual"] = val # Easily modify to export for each value.
            elif self.metrics[metric_name].type == "matrix":
                results[metric_name] = repr(self.metrics[metric_name].value)
            else:
                results[metric_name] = self.metrics[metric_name].value
        return results
     
    def compute(self, data):
        pass

    def update(self, data):
        pass
