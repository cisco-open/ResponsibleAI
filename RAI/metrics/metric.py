__all__ = ['Metric', 'metric_types']

import os.path
from RAI.metrics.registry import register_class

metric_types = {"numeric", "multivalued", "other", "vector", "matrix"}


class Metric:
    def __init__(self, name, config) -> None:
        self.name = name
        self.type = None
        self.explanation = None
        self.value = None
        self.tags = set()
        self.has_range = False
        self.range = None
        self.value_list = None
        self.display_name = None
        self.load_config(config)

    def load_config(self, config):
        if "tags" in config:
            self.tags = config["tags"]
        else:
            self.tags = set()
        if "has_range" in config:
            self.has_range = config["has_range"]
        if "range" in config:
            self.range = config["range"]
        if "explanation" in config:
            self.explanation = config["explanation"]
        self.type = config["type"]
        if "type" in config:
            if config["type"] in metric_types:
                self.type = config["type"]
                print("TYPE: ", self.type)
            else:
                self.type = "numeric"
        if "display_name" in config:
            self.display_name = config["display_name"]
