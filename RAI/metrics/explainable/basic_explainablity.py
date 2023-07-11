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


import os
import numpy as np
from RAI.metrics.metric_group import MetricGroup


class BasicExplainablityGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in \
                    self.ai_system.metric_manager.user_config["stats"]:
                args = self.ai_system.metric_manager.user_config["stats"]["args"]

            scalar_data = data_dict["data"].scalar
            mean_v = np.mean(scalar_data, **args.get("mean", {}), axis=0, keepdims=True)  # noqa: F841
            std_v = np.std(scalar_data, **args.get("covariance", {}), axis=0, keepdims=True)  # noqa: F841
            max_v = np.max(scalar_data, axis=0, keepdims=True)  # noqa: F841
            min_v = np.min(scalar_data, axis=0, keepdims=True)  # noqa: F841

            self.metrics["explainable_model"].value = True

# TODO: This class is a placeholder for Explainability functions. Clarify/remove this class
