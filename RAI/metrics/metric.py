from RAI.utils.all_types import all_metric_types
__all__ = ['Metric']


class Metric:
    """
    Metric class loads in information about a Metric as part of a Metric Group.
    Metrics are automatically created by Metric Groups.
    """

    def __init__(self, name, config) -> None:
        self.config = config
        self.name = name
        self.type = None
        self.explanation = None
        self.value = None
        self.tags = set()
        self.has_range = False
        self.range = None
        self.value_list = None
        self.display_name = None
        self.unique_name = None
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
            if config["type"] in all_metric_types:
                self.type = config["type"]
            else:
                self.type = "numeric"
        if "display_name" in config:
            self.display_name = config["display_name"]
