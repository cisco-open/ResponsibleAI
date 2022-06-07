from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import scipy.stats

# Move config to external .json? 
_config = {
    "name": "frequency_stats",
    "display_name" : "Frequency Statistics Metrics",
    "compatibility": {"type_restriction": None, "output_restriction": None},
    "dependency_list": [],
    "tags": ["stats", "Frequency Stats"],
    "complexity_class": "linear",
    "metrics": {
        "relfreq": {
            "display_name": "Relative Frequency",
            "type": "vector-dict",
            "has_range": True,
            "range": [0, None],
            "explanation": "Indicates the relative count of each subclass.",
        },
        
        "cumfreq": {
            "display_name": "Cumulative Frequency",
            "type": "vector-dict",
            "has_range": True,
            "range": [0, None],
            "explanation": "Indicates the cumulative count of each subclass.",
        },
            }
}

# Type (Regression, Classification, Data | probability, numeric)


class FrequencyStatMetricGroup(MetricGroup, config=_config):
    compatibility = {"type_restriction": None, "output_restriction": None}

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


            # MAKES ASSUMPTION DATA IS FACTORIZED. So categorical variables start at 0 and no numbers are skipped.
            self.metrics["relfreq"].value = _rel_freq(data.X, self.ai_system.meta_database.features)
            self.metrics["cumfreq"].value = _cumulative_freq(data.X, self.ai_system.meta_database.features)
            

def _cumulative_freq(X, features=None):
    result = {}
    for i in range(len(features)):
        if features[i].categorical:
            numbins = len(features[i].values)
            result[features[i].name] = _convert_to_feature_dict(scipy.stats.cumfreq(X[:, i], numbins=numbins)[0].tolist(), features[i]) 
    return result


def _rel_freq(X, features=None):
    result = {}
    for i in range(len(features)):
        if features[i].categorical:
           numbins = len(features[i].values)
           result[features[i].name]  = _convert_to_feature_dict(scipy.stats.relfreq(X[:, i], numbins=numbins)[0].tolist(), features[i])
    return result


def _convert_to_feature_dict(values, feature):
    result = {}
    for i in range(len(values)):
        result[feature.values[i]] = values[i]
    return result



# Old functions, may reuse.
def _calculate_per_feature(function, X, feature):
    result = {}
    for value in feature.values:
        result[feature.values[value]] = function(X, value)
    return result



def _calculate_per_categorical_value(function, X, features=None, per_feature=False):
    result = []
    for i in range(len(features)):
        if not features[i].categorical:
            result.append(None)
        else:
            if per_feature:
                result.append(_calculate_per_feature(function, X[:, i], features[i]))
            else:
                result.append(function(X[:, i]))
                print("DATA: ")
                print(X[:, i])
    return result


def _get_scalar_data(X, mask):
    result = np.copy(X)
    i = len(mask)-1
    while i >= 0:
        if mask[i] == 1:
            result = np.delete(result, i, axis=1)
        i = i-1
    return result