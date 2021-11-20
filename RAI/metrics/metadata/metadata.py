from RAI.metrics.metric_group import MetricGroup
import datetime



_config = {
    "name": "metadata",
    "compatibility": {"type_restriction": None, "output_restriction": None},
    "dependency_list": [],
    "tags": ["metadata"],
    "complexity_class": None,
    "metrics": {
        "date": {
            "display_name": "Date",
            "type": "text",
            "has_range": False,
            "range": [None, None],
            "explanation": "The Date in which a measurement was taken.",
        },
        "description": {
            "display_name": "Measurement Description",
            "type": "text",
            "has_range": False,
            "range": [None, None],
            "explanation": "The user description of collected metric data.",
        }, 
    }
}


class MetadataGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        self.metrics["date"].value = self._get_time()
        self.metrics["description"].value = ""

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)

