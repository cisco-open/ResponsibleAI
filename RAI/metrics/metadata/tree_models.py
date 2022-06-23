from RAI.metrics.metric_group import MetricGroup
import datetime
import os


class TreeModels(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    @classmethod
    def is_compatible(cls, ai_system):
        compatible = super().is_compatible(ai_system)
        return compatible and ai_system.model.agent.__class__.__module__.split(".")[0] == "sklearn"

    def compute(self, data_dict):
        model = self.ai_system.model.agent
        self.metrics["estimator_counts"].value = None
        self.metrics["estimator_params"].value = None
        self.metrics["feature_names"].value = None
        if hasattr(model, 'n_estimators'):
            self.metrics["estimator_counts"].value = model.n_estimators
        if hasattr(model, 'estimators_'):
            self.metrics["estimator_params"].value = model.estimators_
        self.metrics["feature_names"].value = [f.name for f in self.ai_system.meta_database.features]

    # TODO: Does not work with Decision Trees

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)

