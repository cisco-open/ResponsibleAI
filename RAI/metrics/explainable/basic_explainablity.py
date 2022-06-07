from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np


# Move config to external .json? 
_config = {
    "name" : "basic_explainablity",
    "display_name" : "Basic Robustness Metrics",
    "compatibility": {"type_restriction": None, "output_restriction": None},
    "dependency_list": [],
    "tags": ["robustness", "Normalization"],
    "complexity_class": "linear",
    "metrics": {
        "explainable_model": {
            "display_name": "explainable model",
            "type": "boolean",
            "has_range": False,
            "range": [None, None],
            "explanation": "Whether of not model is explainable",
        },
       
    }
}

# Type (Regression, Classification, Data | probability, numeric)


class BasicExplainablityGroup(MetricGroup, config = _config):
     
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

            scalar_data = data.X[:,self.ai_system.meta_database.scalar_mask]

            mean_v = np.mean(scalar_data, **args.get("mean", {}), axis=0, keepdims=True)
            std_v = np.std(scalar_data, **args.get("covariance", {}), axis=0, keepdims= True )
            max_v = np.max(scalar_data, axis=0, keepdims=True)
            min_v = np.min(scalar_data, axis=0, keepdims=True)


            self.metrics["explainable_model"].value =  True
            
            # bool(np.all(np.isclose(max_v, np.ones_like(max_v))) and \
            #                                         np.all(np.isclose(min_v, np.zeros_like(min_v))))



