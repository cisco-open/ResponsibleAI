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


from RAI.all_types import all_metric_types
__all__ = ['Metric']


class Metric:
    """
    Metric class loads in information about a Metric as part of a Metric Group.
    Metrics are automatically created by Metric Groups.
    """

    def __init__(self, name, config) -> None:
        self.config = config
        self.name = name
        self.type = None
        self.explanation = None
        self.value = None
        self.tags = set()
        self.has_range = False
        self.range = None
        self.value_list = None
        self.display_name = None
        self.unique_name = None
        self.load_config(config)

    def load_config(self, config):
        """
        loads the config details consisting of tags, has_range, range, explanation, type and display_name

        :param config: Config details

        :return: None

        """
        if "tags" in config:
            self.tags = config["tags"]
        else:
            self.tags = set()
        if "has_range" in config:
            self.has_range = config["has_range"]
        if "range" in config:
            self.range = config["range"]
        if "explanation" in config:
            self.explanation = config["explanation"]
        self.type = config["type"]
        if "type" in config:
            if config["type"] in all_metric_types:
                self.type = config["type"]
            else:
                self.type = "numeric"
        if "display_name" in config:
            self.display_name = config["display_name"]
