__all__ = ['Metric', 'metric_types']

import os.path
from RAI.metrics.registry import register_class

metric_types = {  "numeric" , "multivalued", "other"}


class Metric:
    def __init__(self, name, config) -> None:
        self.name = name
        self.value = None
        self.tags = set()
        self.has_range = False
        self.range = None
        self.value_list = None
        self.load_config(config)
        
    
    def load_config(self, config):
        if "type" in config:
            if config["type"] in metric_types:
                self.type = config["type"]
            else:
                self.type = "numeric"


        if "tags" in config:
            self.tags = config["tags"]
        else:
            self.tags = set()


        #load other attributes







