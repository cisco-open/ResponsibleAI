from .metric_group import MetricGroup
from RAI.metrics.registry import registry
from RAI import utils
 


__all__ = ['MetricManager']

import json
import os.path
import site

# choose the first site packages folder
site_pkgs_path = site.getsitepackages()[0]
rai_pkg_path = os.path.join(site_pkgs_path, "RAI")
if not os.path.isdir(rai_pkg_path):
    rai_pkg_path = "RAI"


class MetricManager(object):

    def __init__(self, ai_system) -> None:
        super().__init__()
        self.ai_system = ai_system
        self.metric_groups = {}
        self.user_config = {"fairness": {   "priv_group": {},
                                            "protected_attributes": [], "positive_label": 1},
                                            "time_complexity": "exponential"}

    def standardizeUserConfig(self, user_config: dict):
        if "fairness" in user_config:
            protected_classes = []
            if "priv_group" in user_config["fairness"]:
                for attr in user_config["fairness"]["priv_group"]:
                    protected_classes.append(attr)
                    assert "privileged" in user_config["fairness"]["priv_group"][attr]
                    assert "unprivileged" in user_config["fairness"]["priv_group"][attr]
                assert "positive_label" in user_config["fairness"]
                user_config["fairness"]["protected_attributes"] = protected_classes
                print("protected attributes: ", protected_classes)

    def initialize(self, user_config: dict = None, metric_groups: list[str] = None, max_complexity: str = "linear"):
        if user_config:
            self.standardizeUserConfig(user_config)
            for key in user_config:
                self.user_config[key] = user_config[key]

        compatible_metrics = []  # Stores compatible metrics
        dependencies = {}  # Stores a metrics dependencies
        dependent = {}  # Maps metrics to metrics dependent on it

        whitelist = user_config.get("whitelist", registry)
        blacklist = user_config.get("blacklist", [])

        # Find all compatible metric groups
        for metric_group_name in registry:
            if metric_groups is not None and metric_group_name not in metric_groups:
                continue
            metric_class = registry[metric_group_name]
            if metric_class.is_compatible(self.ai_system) and metric_group_name in whitelist and metric_group_name not in blacklist:
                compatible_metrics.append(metric_class)
                dependencies[metric_class.config["name"]] = metric_class.config["dependency_list"]
                for dependency in metric_class.config["dependency_list"]:
                    if dependent.get(dependency) is None:
                        dependent[dependency] = []
                    dependent[dependency].append(metric_class.config["name"])

        # Remove metrics with missing dependencies
        removed = True
        while removed:
            removed = False
            for metric in compatible_metrics:
                for metric_dependency in dependencies[metric.config["name"]]:
                    if metric_dependency not in dependencies:
                        compatible_metrics.remove(metric)
                        dependencies.pop(metric.config["name"])
                        print("Missing dependency ", metric_dependency, " for ", metric.config["name"])
                        removed = True
                        break

        # Check for circular dependencies
        while len(compatible_metrics) != 0:
            removed = False
            for metric in compatible_metrics:
                metric_name = metric.config["name"]
                if len(dependencies[metric_name]) == 0:
                    for dependent_metric in dependent.get(metric_name, []):
                        dependencies[dependent_metric].remove(metric_name)
                    self.metric_groups[metric_name] = metric(self.ai_system)
                    print(f"metric group : {metric_name} was loaded")
                    compatible_metrics.remove(metric)
                    removed = True
            if not removed:
                raise AttributeError("Circular dependency detected in ", [val.name for val in compatible_metrics])

    def reset_measurements(self) -> None:
        for metric_group_name in self.metric_groups:
           self.metric_groups[metric_group_name].reset()

        self._last_certificate_values = None
        self._last_metric_values = None
        self._sample_count = 0
        self._time_stamp = None  # Replace by registering a time metric in metric_groups?       

    def get_metadata(self) -> dict :
        result = {}
        for group in self.metric_groups:
            result[group] = {}
            result[group]["meta"] = {
                "tags": self.metric_groups[group].tags,
                "complexity_class":self.metric_groups[group].complexity_class,
                "dependency_list": self.metric_groups[group].dependency_list,
                "compatiblity": self.metric_groups[group].compatiblity,
                "display_name": self.metric_groups[group].display_name
            }
            for metric in self.metric_groups[group].metrics:
                result[group][metric] = self.metric_groups[group].metrics[metric].config
        return result

    def get_metric_info_flat(self) -> dict :
        result = {}
        for group in self.metric_groups:
             
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[metric_obj.unique_name] = metric_obj.config
                metric_obj.config["tags"] = self.metric_groups[group].tags  # Change this up after
        return result
    
    def compute(self, data_dict) -> dict :
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].compute(data_dict)

        result = {}
        for group in self.metric_groups:
            result[group] = {}
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[group][metric] =  utils.jsonify(metric_obj.value)
        return result

    # Searches all metrics. Queries based on Metric Name, Metric Group Name, Category, and Tags.
    def search(self, query:str) -> dict :
        query = query.lower()
        results = {}
        for group in self.metric_groups:
            add_group = group.lower() == query 
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                if add_group or metric.lower().find(query) > -1 or metric_obj.display_name.lower().find(query) > -1:
                    results[metric] = metric_obj.value
                elif metric_obj.tags is not None:
                    for tag in metric_obj.tags:
                        if tag.lower().find(query) > -1:
                            results[metric] = metric_obj.value
                            break
        return results
