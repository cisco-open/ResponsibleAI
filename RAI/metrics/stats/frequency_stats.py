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


from RAI.metrics.metric_group import MetricGroup
import scipy.stats
from RAI.utils.utils import convert_to_feature_value_dict
import os


class FrequencyStatMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def compute(self, data_dict):
        # args = {}
        # if self.ai_system.metric_manager.user_config is not None \
        #         and "stats" in self.ai_system.metric_manager.user_config \
        #         and "args" in self.ai_system.metric_manager.user_config["stats"]:
        #     args = self.ai_system.metric_manager.user_config["stats"]["args"]
        data = data_dict["data"]
        self.metrics["relative_freq"].value = _rel_freq(data.X, self.ai_system.meta_database.features)
        self.metrics["cumulative_freq"].value = _cumulative_freq(data.X, self.ai_system.meta_database.features)


def _cumulative_freq(X, features=None):
    result = {}
    for i in range(len(features)):
        if features[i].categorical:
            numbins = len(features[i].values)
            result[features[i].name] = convert_to_feature_value_dict(
                scipy.stats.cumfreq(X[:, i], numbins=numbins)[0].tolist(), features[i]
            )
    return result


def _rel_freq(X, features=None):
    result = {}
    for i in range(len(features)):
        if features[i].categorical:
            numbins = len(features[i].values)
            result[features[i].name] = convert_to_feature_value_dict(
                scipy.stats.relfreq(X[:, i], numbins=numbins)[0], features[i])
    return result
