# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from typing import List
from RAI import utils
from RAI.all_types import all_output_requirements, all_complexity_classes, all_dataset_requirements, \
    all_data_types, all_task_types
from RAI.metrics.metric_registry import registry
import logging
logger = logging.getLogger(__name__)

__all__ = ['MetricManager']

# choose the first site packages folder
# site_pkgs_path = site.getsitepackages()[0]
# rai_pkg_path = os.path.join(site_pkgs_path, "RAI")
# if not os.path.isdir(rai_pkg_path):
rai_pkg_path = "RAI"


class MetricManager(object):
    """
    MetricManager is used to create and Manage various MetricGroups which are compatible with the
    AISystem. MetricManager is created by the AISystem, and will load in all available MetricGroups compatible
    with the AISystem. MetricManager also provides functions to run computes for all metric groups,
    get metadata about metric groups, and get metric values.
    """

    def __init__(self, ai_system) -> None:
        super().__init__()
        self._time_stamp = None
        self._sample_count = 0
        self._last_metric_values = None
        self._last_certificate_values = None
        self.ai_system = ai_system
        self.metric_groups = {}
        self.user_config = {"fairness": {"priv_group": {}, "protected_attributes": [], "positive_label": 1},
                            "time_complexity": "exponential"}

    def standardize_user_config(self, user_config: dict):
        """
        Accepts user config values and make in standard group

        :param user_config(dict): user config data
        :return: None
        """
        if "fairness" in user_config:
            protected_classes = []
            if "priv_group" in user_config["fairness"]:
                for attr in user_config["fairness"]["priv_group"]:
                    protected_classes.append(attr)
                    assert "privileged" in user_config["fairness"]["priv_group"][attr]
                    assert "unprivileged" in user_config["fairness"]["priv_group"][attr]
                assert "positive_label" in user_config["fairness"]
                user_config["fairness"]["protected_attributes"] = protected_classes

    def initialize(self, user_config: dict = None, metric_groups: List[str] = None, max_complexity: str = "linear"):
        """
        Find all compatible metric groups and Remove metrics with missing dependencies and Check for circular dependencies

        :param user_config(dict): user config data
        :param metric_groups: metric groups data as a list
        :param max_complexity: default linear

        :return: None
        """

        if user_config:
            self.standardize_user_config(user_config)
            for key in user_config:
                self.user_config[key] = user_config[key]

        self.metric_groups = {}
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
            self._validate_config(metric_class.config)
            if metric_class.is_compatible(
                    self.ai_system) and metric_group_name in whitelist and metric_group_name not in blacklist:
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
                    logger.info(f"metric group: {metric_name} was loaded")
                    compatible_metrics.remove(metric)
                    removed = True
            if not removed:
                raise AttributeError("Circular dependency detected in ", [val.name for val in compatible_metrics])

    def reset_measurements(self) -> None:
        """
        Reset all the certificate, metric, sample and time_stamp values

        :param self: None

        :return: None

        """
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].reset()

        self._last_certificate_values = None
        self._last_metric_values = None
        self._sample_count = 0
        self._time_stamp = None

    def get_metadata(self) -> dict:
        """
        Return the metric group metadata information

        :param self: None

        :return: dict-Metadata

        """
        result = {}
        for group in self.metric_groups:
            result[group] = {}
            result[group]["meta"] = {
                "tags": self.metric_groups[group].tags,
                "complexity_class": self.metric_groups[group].complexity_class,
                "dependency_list": self.metric_groups[group].dependency_list,
                "compatiblity": self.metric_groups[group].compatiblity,
                "display_name": self.metric_groups[group].display_name
            }
            for metric in self.metric_groups[group].metrics:
                result[group][metric] = self.metric_groups[group].metrics[metric].config
        return result

    def get_metric_info_flat(self) -> dict:
        """
        Returns the metric info

        :param self: None

        :return: Returns the metric info data in dict
        """
        result = {}
        for group in self.metric_groups:

            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[metric_obj.unique_name] = metric_obj.config
                metric_obj.config["tags"] = self.metric_groups[group].tags  # Change this up after
        return result

    def compute(self, data_dict) -> dict:
        """
        Perform computation on metric objects and returns the value as a metric group in dict format

        :param data_dict: Accepts the data dict metric object

        :return: returns the value as a metric group
        """
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].compute(data_dict)
        result = {}
        for group in self.metric_groups:
            result[group] = {}
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[group][metric] = utils.jsonify(metric_obj.value)
        return result

    def iterator_compute(self, data_dict, preds: dict) -> dict:
        """
        Accepts data_dict and preds as a input and returns the metric objects from a batch of metric group

        :param data_dict: Accepts the data dict metric object
        :param  preds: prediction value from the detection
        :return: returns the metric objects from a batch of metric group
        """
        data = data_dict["data"]
        data.reset()
        cur_idx = 0
        # Need to reset metric group values
        for group in self.metric_groups:
            self.metric_groups[group].reset()

        while data.next_batch():
            data_len = 0
            if data.X is not None:
                data_len = len(data.X)
            elif data.y is not None:
                data_len = len(data.y)

            for output_type in all_output_requirements:
                if output_type in data_dict:
                    data_dict[output_type] = preds[output_type][cur_idx: cur_idx + data_len]
                elif output_type in data_dict:
                    data_dict.pop(output_type)
            cur_idx += data_len

            for metric_group_name in self.metric_groups:
                self.metric_groups[metric_group_name].compute_batch(data_dict)

        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].finalize_batch_compute()

        result = {}
        for group in self.metric_groups:
            result[group] = {}
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[group][metric] = utils.jsonify(metric_obj.value)
        return result

    # batched_compute
    # if data instance of IteratorData, iterate through batches,

    # Searches all metrics. Queries based on Metric Name, Metric Group Name, Category, and Tags.
    def search(self, query: str) -> dict:
        """
        Searches all metrics.Queries based on Metric Name, Metric Group Name, Category, and Tags

        :param query: query(str) group data information as input

        :return:  Returns the search results based on the query

        """
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

    def _validate_config(self, config):
        assert "name" in config and isinstance(config["name"], str), \
            "All configs must contain names"

        assert "display_name" in config and isinstance(config["display_name"], str), \
            config["name"] + " must contain a valid display name"

        assert "compatibility" in config, \
            config["name"] + " must contain compatibility details"

        assert "task_type" in config["compatibility"] and (config["compatibility"]["task_type"] == [] or  # noqa : W504
               all(i in all_task_types for i in config["compatibility"]["task_type"])), \
            config["name"] + "['compatibility']['task_type'] must be empty or one of " + str(all_task_types)  # noqa : E128

        assert "data_type" in config["compatibility"] and all(x in all_data_types for x in config["compatibility"]["data_type"]), \
            config["name"] + "['compatibility']['data_type'] must be one of " + str(all_data_types)

        assert "output_requirements" in config["compatibility"] and \
               all(x in all_output_requirements for x in config["compatibility"]["output_requirements"]), \
            config["name"] + "['compatibility']['output_requirements'] must be one of " + str(all_output_requirements)

        assert "dataset_requirements" in config["compatibility"] and \
               all(x in all_dataset_requirements for x in config["compatibility"]["dataset_requirements"]), \
            config["name"] + "['compatibility']['dataset_requirements'] must be one of " + str(all_dataset_requirements)

        assert "dependency_list" in config and isinstance(config["dependency_list"], list) and \
               all(isinstance(x, str) for x in config["dependency_list"]), \
            config["name"] + " must contain a dependency list"

        assert "tags" in config and isinstance(config["tags"], list) and \
               all(isinstance(x, str) for x in config["tags"]), \
            config["name"] + " must contain a list of 0 or more string tags"

        assert "complexity_class" in config and config["complexity_class"] in all_complexity_classes, \
            config["name"] + " must have a complexity class belong to " + str(all_complexity_classes)

        assert "metrics" in config, \
            config["name"] + " must contain metrics."

        for metric in config["metrics"]:
            assert "display_name" in config["metrics"][metric] and \
                   isinstance(config["metrics"][metric]["display_name"], str), \
                metric + " must have a valid display name."

            assert "type" in config["metrics"][metric] and isinstance(config["metrics"][metric]["type"], str), \
                metric + " must contain a valid type."

            assert "has_range" in config["metrics"][metric] and \
                   isinstance(config["metrics"][metric]["has_range"], bool), \
                metric + " must contain a boolean for has_range."

            assert "range" in config["metrics"][metric] and (
                config["metrics"][metric]["range"] is None or (
                    isinstance(config["metrics"][metric]["range"], list) and  # noqa : W504
                    len(config["metrics"][metric]["range"]) == 2 and (
                        x in {None, False, True} for x in config["metrics"][metric]["range"]))), \
                metric + " must contain a valid list of length 2 consisting of null, false or true."

            assert "explanation" in config["metrics"][metric] and \
                   isinstance(config["metrics"][metric]["explanation"], str), \
                metric + " must contain a valid explanation."
