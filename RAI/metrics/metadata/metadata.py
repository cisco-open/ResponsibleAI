from RAI.metrics.metric_group import MetricGroup
import datetime
import os


class MetadataGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        self.metrics["date"].value = self._get_time()
        self.metrics["description"].value = self.ai_system.model.description
        self.metrics["sample_count"].value = data_dict["data"].X.shape[0]
        self.metrics["task_type"].value = self.ai_system.model.task
        if self.ai_system.model.agent:
            self.metrics["model"].value = str(self.ai_system.model.agent)
        else:
            self.metrics["model"].value = "None"

        self.metrics["tag"].value = data_dict["tag"]

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)

