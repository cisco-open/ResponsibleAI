from RAI.metrics.metric_group import MetricGroup
import datetime



_config = {
    "name": "Tree Models",
    "display_name" : "SKlearn Tree Based Models",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "dependency_list": [],
    "tags": ["metadata"],
    "complexity_class": "linear",
    "metrics": {
        "estimator_counts": {
            "display_name": "Number of estimators",
            "type": "numeric",
            "has_range": True,
            "range": [1, None],
            "explanation": "Number of estimators",
        },
        "estimator_params": {
            "display_name": "estimators data",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "estimators data",
        }, 
        "feature_names": {
            "display_name": "feature names",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "feature names",
        }, 
    }
}


class TreeModels(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass
    
    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible and ai_system.task.model.agent.__class__.__module__.split(".")[0] == "sklearn"  

        return compatible

    def compute(self, data_dict):
        model = self.ai_system.task.model.agent
        self.metrics["estimator_counts"].value = model.n_estimators
        self.metrics["estimator_params"].value = model.estimators_
        self.metrics["feature_names"].value = [f.name for f in self.ai_system.meta_database.features]

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)

