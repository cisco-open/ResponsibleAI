from .metric import Metric
from RAI.metrics.registry import register_class
import numpy as np

__all__ = ['MetricGroup']

all_complexity_classes = {"constant",  "linear",  "multi_linear", "polynomial", "exponential"}


class MetricGroup(object):    
    name = ""

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = name
        register_class(name, cls)

    def __init__(self, ai_system, config) -> None:
        self.ai_system = ai_system
        self.persistent_data = {}
        self.dependency_list = []
        self.metrics = {}
        self.tags = []
        self.category = None
        self.complexity_class = None
        self.status = "OK"
        self.reset()
        
        if self.load_config(config):
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
            self.complity_class = config["complexity_class"]
        if "metrics" in config:
            self.create_metrics(config["metrics"])
        if "category" in config:
            self.category = config["category"]

    def create_metrics(self, metrics_config):
        for metric_name in metrics_config:
            self.metrics[metric_name] = Metric(metric_name, metrics_config[metric_name])

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


