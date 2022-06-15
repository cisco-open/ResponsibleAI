from RAI.metrics.metric_group import MetricGroup
import datetime



_config = {
    "name": "metadata",
    "display_name" : "Metadata for the measurement",
    "compatibility": {"type_restriction": None, "output_restriction": None},
    "dependency_list": [],
    "tags": ["metadata"],
    "complexity_class": "linear",
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
         "sample_count": {
            "display_name": "Number of samples",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Number of samples",
        }, 
        "task_type": {
            "display_name": "Task Type",
            "type": "text",
            "has_range": False,
            "range": [None, None],
            "explanation": "Task Type",
        },
        "model": {
            "display_name": "model",
            "type": "text",
            "has_range": False,
            "range": [None, None],
            "explanation": "model description",
        },
         "tag": {
            "display_name": "tag",
            "type": "text",
            "has_range": False,
            "range": [None, None],
            "explanation": "measurement tag",
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
        self.metrics["description"].value = self.ai_system.task.description
        self.metrics["sample_count"].value = data_dict["data"].X.shape[0]
        self.metrics["task_type"].value = self.ai_system.task.type
        if self.ai_system.task.model: 
            self.metrics["model"].value = str(self.ai_system.task.model.agent)
        else:
            self.metrics["model"].value = "None"

        self.metrics["tag"].value = data_dict["tag"]

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)

