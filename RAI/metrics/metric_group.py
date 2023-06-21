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


from abc import ABC, abstractmethod
import json
import numpy as np
from RAI.metrics.metric_registry import register_class
from RAI.utils import compare_runtimes
from .metric import Metric

__all__ = ['MetricGroup']


class MetricGroup(ABC):
    """
    MetricGroups are a group of related metrics. This class loads in information about a
    metric group from its .json file. This class then creates associated metrics for the group,
    provides compatibility checking, run computes. Metric Groups are created by MetricManagers.
    """

    name = ""
    config = None

    # Checks if the group is compatible with the provided AiSystem
    @classmethod
    def is_compatible(cls, ai_system):

        """
        Checks if the group is compatible with the provided AiSystem


        :param: ai_system

        :return: Compatible object

        """

        compatible = cls.config["compatibility"]["task_type"] == []\
            or any(i == ai_system.task for i in cls.config["compatibility"]["task_type"])\
            or (any(i == "classification" for i in cls.config["compatibility"]["task_type"])
                and ai_system.task == "binary_classification")  # noqa : W503
        compatible = compatible and (cls.config["compatibility"]["data_type"] is None or cls.config["compatibility"][
            "data_type"] == [] or all(item in ai_system.meta_database.data_format
                                      for item in cls.config["compatibility"]["data_type"]))
        compatible = compatible and (cls.config["compatibility"]["output_requirements"] == [] or  # noqa : W504
                                     all(item in ai_system.data_dict for item in  # noqa : W504
                                         cls.config["compatibility"]["output_requirements"]))  # noqa : W504
        compatible = compatible and (cls.config["compatibility"]["dataset_requirements"] is None or  # noqa : W504
                                     all(item in ai_system.meta_database.stored_data for item in
                                         cls.config["compatibility"]["dataset_requirements"]))
        compatible = compatible and (cls.config["compatibility"]["data_requirements"] == [] or  # noqa : W504
                                     all(type(item).__name__ in cls.config["compatibility"]["data_requirements"] for
                                         item in ai_system.dataset.data_dict.values()))
        compatible = compatible and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"),
                                                     cls.config["complexity_class"])
        return compatible

    # Registers a subclass
    def __init_subclass__(cls, class_location=None, **kwargs):
        super().__init_subclass__(**kwargs)
        config_file = class_location[:-2] + "json"
        cls.config = json.load(open(config_file))
        cls.name = cls.config["name"]
        register_class(cls.name, cls)

    def __init__(self, ai_system) -> None:
        self.ai_system = ai_system
        self.persistent_data = {}
        self.dependency_list = []
        self.metrics = {}
        self.tags = []
        self.complexity_class = ""
        self.display_name = self.name
        self.compatiblity = {}
        self.status = "OK"
        self.reset()

        if self.load_config(self.config):
            self.status = "OK"
        else:
            self.status = "BAD"

    def reset(self):
        """
        Reset the status

        :param: None

        :return: None

        """
        if self.status == "BAD":
            return
        self.persistent_data = {}
        for metric in self.metrics:
            self.metrics[metric].value = None
        self.status = "OK"

    def load_config(self, config):
        """
        Fetch the metric data from config

        :param: config

        :return: Boolean

        """
        if "tags" in config:
            self.tags = config["tags"]
        if "dependency_list" in config:
            self.dependency_list = config["dependency_list"]
        if "complexity_class" in config:
            self.complexity_class = config["complexity_class"]
        if "compatibility" in config:
            self.compatiblity = config["compatibility"]
        if "display_name" in config:
            self.display_name = config["display_name"]
        else:
            self.display_name = self.name
        if "metrics" in config:
            self.create_metrics(config["metrics"])
        return True

    def create_metrics(self, metrics_config):
        """
        Create the metric and assign name and tags to it

        :param: metrics_config

        :return: None

        """
        for metric_name in metrics_config:
            self.metrics[metric_name] = Metric(metric_name, metrics_config[metric_name])
            self.metrics[metric_name].unique_name = self.name + " > " + metric_name
            self.metrics[metric_name].tags = self.tags

    def get_metric_values(self):
        """
        Returns the metric with the name and its corresponding value

        :param: None

        :return: dict

        """
        results = {}
        for metric_name in self.metrics:
            if self.metrics[metric_name].type == 'vector':
                results[metric_name + "-single"] = self.metrics[metric_name].value[0]
                val = self.metrics[metric_name].value[1]
                if type(self.metrics[metric_name].value[1]) is np.ndarray:
                    val = val.tolist()
                results[metric_name + "-individual"] = val  # Easily modify to export for each value.
            elif self.metrics[metric_name].type == 'feature-array':
                val = val = self.metrics[metric_name].value
                if type(self.metrics[metric_name].value) is np.ndarray:
                    val = val.tolist()
                results[metric_name] = val
            else:
                results[metric_name] = self.metrics[metric_name].value
        return results

    def export_metric_values(self):
        """
        Returns the metric with the name and its corresponding value

        :param: None

        :return: dict

        """
        results = {}
        for metric_name in self.metrics:
            if self.metrics[metric_name].type == 'vector':
                results[metric_name + "-single"] = self.metrics[metric_name].value[0]
                val = self.metrics[metric_name].value[1]
                if type(self.metrics[metric_name].value[1]) is np.ndarray:
                    val = val.tolist()
                results[metric_name + "-individual"] = val  # Easily modify to export for each value.
            elif self.metrics[metric_name].type == "matrix":
                results[metric_name] = repr(self.metrics[metric_name].value)
            else:
                results[metric_name] = self.metrics[metric_name].value
        return results

    @abstractmethod
    def compute(self, data):
        pass

    def compute_batch(self, data):
        pass

    def finalize_batch_compute(self):
        pass

    def update(self, data):
        pass
