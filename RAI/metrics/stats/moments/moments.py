from ..stats import StatMetricGroup
import math

_config = {
    "src": "moments",
    "dependency_list": ["stat"],
    "tags": ["moments"],
    "complexity_class": "linear",
    "metrics": {
        "second-moment": {
            "type": "other",
            "tags": [],
            "has_range": False,

        },
        "third-moment": {
            "type": "other",
            "tags": [],
            "has_range": False,
        },
 
    }

}


class MomentMetricGroup(StatMetricGroup, name="moment"):
    def __init__(self, ai_system, config=_config) -> None:
        super().__init__(ai_system, config)

    def update(self, data):
        pass
    
    def compute(self, data_dict):
        data = data_dict["data"]
        # these are just for feature test, not really correct
        self.metrics["second-moment"].value = 2* self.ai_system.get_metric("stat", "mean")
        self.metrics["third-moment"].value = 3* 2* self.ai_system.get_metric("stat", "mean")
