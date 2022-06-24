from RAI.metrics.metric_group import MetricGroup
import scipy.stats
import os


class StatMomentGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
                args = self.ai_system.metric_manager.user_config["stats"]["args"]
            data = data_dict["data"]
            scalar_data = data.scalar

            self.metrics["moment_1"].value = scipy.stats.moment(scalar_data, 1)
            self.metrics["moment_2"].value = scipy.stats.moment(scalar_data, 2)
            self.metrics["moment_3"].value = scipy.stats.moment(scalar_data, 3)
 