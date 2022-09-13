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


"""config parse module."""

import yaml
from RAI.all_types import all_task_types
from ..dataset import Feature

KEY_TASK_TYPE = 'taskType'
KEY_FEATURES = 'features'
KEY_USER_CONFIG = 'userConfig'

# keys in the feature section
KEY_FEATURE_NAME = 'name'
KEY_FEATURE_TYPE = 'type'
KEY_FEATURE_DESC = 'description'
KEY_FEATURE_CATE = 'categorical'
KEY_FEATURE_VALUES = 'values'


class Config(object):
    """RAI configuration parsing class."""

    def __init__(self, config_path: str) -> None:
        """Initialize an instance."""
        self.config_path = config_path

        self.task_type = ""
        self.features = list()
        self.user_config = {}

        self._parse()

    def __str__(self):
        """Override __str__ dunder function."""
        task_type = f"task_type: {self.task_type}"
        user_config = f"user config: {self.user_config}"

        features = list()
        for idx, feature in enumerate(self.features):
            feature_str = f"feature {idx}: {feature}"
            features.append(feature_str)

        features_str = '\n'.join(features)

        return '\n'.join([task_type, features_str, user_config])

    def get(self):
        """Return configuration objects from config.

        Returns
        -------
        task_type: type of task (defined in AISystem.task)
        features: an array of feature objects
        user_config: user-defined config
        """
        return self.task_type, self.features, self.user_config

    def _parse(self):
        """Parse RAI configuration."""
        with open(self.config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

            self._parse_task_type(yaml_data)

            self._parse_features(yaml_data)

            self._parse_user_config(yaml_data)

    def _parse_task_type(self, yaml_data) -> None:
        if KEY_TASK_TYPE not in yaml_data:
            raise KeyError("task type not found")

        self.task_type = yaml_data[KEY_TASK_TYPE]

        if self.task_type not in all_task_types:
            raise ValueError(f"unkown task type: {self.task_type}")

    def _parse_features(self, yaml_data) -> None:
        if KEY_FEATURES not in yaml_data:
            return

        for feature_data in yaml_data[KEY_FEATURES]:
            name = ""
            dtype = ""
            desc = ""
            categorical = False
            values = None

            if KEY_FEATURE_NAME in feature_data:
                name = feature_data[KEY_FEATURE_NAME]
            if KEY_FEATURE_TYPE in feature_data:
                dtype = feature_data[KEY_FEATURE_TYPE]
            if KEY_FEATURE_DESC in feature_data:
                desc = feature_data[KEY_FEATURE_DESC]
            if KEY_FEATURE_CATE in feature_data:
                categorical = feature_data[KEY_FEATURE_CATE]
            if KEY_FEATURE_VALUES in feature_data:
                values = feature_data[KEY_FEATURE_VALUES]

            if not name or not dtype:
                key_name, key_type = KEY_FEATURE_NAME, KEY_FEATURE_TYPE
                raise ValueError(
                    f"{key_name} and {key_type} must not be empty")

            feature = Feature(name, dtype, desc, categorical, values)
            self.features.append(feature)

    def _parse_user_config(self, yaml_data) -> None:
        if KEY_USER_CONFIG not in yaml_data:
            return

        self.user_config = yaml_data[KEY_USER_CONFIG]
