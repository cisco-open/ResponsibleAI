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


from RAI.AISystem import AISystem
from abc import ABC, abstractmethod
from .analysis_registry import register_class
from RAI.utils import compare_runtimes
import json


class Analysis(ABC):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        self.ai_system = ai_system
        self.analysis = {}
        self.dataset = dataset
        self.tag = None
        self.max_progress_tick = 5
        self.current_tick = 0
        self.connection = None  # connection is an optional function passed to share progress with dashboard
        print("Analysis created")

    def __init_subclass__(cls, class_location=None, **kwargs):
        super().__init_subclass__(**kwargs)
        config_file = class_location[:-2] + "json"
        cls.config = json.load(open(config_file))
        register_class(cls.__name__, cls)

    @classmethod
    def is_compatible(cls, ai_system, dataset: str):
        """
        :param ai_system: input the ai_system object
        :param dataset: input the dataset

        :return: class object

        Returns the classifier and sklearn object data
        """
        compatible = cls.config["compatibility"]["task_type"] is None \
            or cls.config["compatibility"]["task_type"] == [] \
            or ai_system.task in cls.config["compatibility"]["task_type"] \
            or ("classification" in cls.config["compatibility"]["task_type"]
                and ai_system.task == "binary_classification")  # noqa: W503
        compatible = compatible and (cls.config["compatibility"]["data_type"] is None or cls.config["compatibility"][
            "data_type"] == [] or all(item in ai_system.meta_database.data_format
                                      for item in cls.config["compatibility"]["data_type"]))
        compatible = compatible and (cls.config["compatibility"]["output_requirements"] is None or  # noqa: W503, W504
                                     all(item in ai_system.data_dict for item in
                                         cls.config["compatibility"]["output_requirements"]))  # noqa: W503, W504
        compatible = compatible and (cls.config["compatibility"]["dataset_requirements"] is None or  # noqa: W503, W504
                                     all(item in ai_system.meta_database.stored_data for item in
                                         cls.config["compatibility"]["dataset_requirements"]))  # noqa: W503, W504
        compatible = compatible and (cls.config["compatibility"]["data_requirements"] == [] or  # noqa: W503, W504
                                     all(type(item).__name__ in cls.config["compatibility"]["data_requirements"] for
                                         item in ai_system.dataset.data_dict.values()))
        compatible = compatible and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"),
                                                     cls.config["complexity_class"])
        compatible = compatible and all(group in ai_system.get_metric_values()[dataset]
                                        for group in cls.config["compatibility"]["required_groups"])
        return compatible

    def progress_percent(self, percentage_complete):
        """
        :parameter: percentage_complete


        :return: None

        Shows the progress percent value
        """
        percentage_complete = int(percentage_complete)
        if self.conncetion is not None:
            self.connection(str(percentage_complete))

    def progress_tick(self):
        """
        :parameter: None


        :return: None

        On every compute it changes the current_tick value

        """
        self.current_tick += 1
        percentage_complete = min(100, int(100 * self.current_tick / self.max_progress_tick))
        if self.connection is not None:
            self.connection(str(percentage_complete))

    # connection is a function that accepts progress, and pings the dashboard
    def set_connection(self, connection):
        """
        :param connection: inputs connection data


        :return: None

        Connection is a function that accepts progress, and pings the dashboard

        """
        self.connection = connection

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def to_html(self):
        pass
