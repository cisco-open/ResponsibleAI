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
import sklearn
import os


class PerformanceRegMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def compute(self, data_dict):
        data = data_dict["data"]
        preds = data_dict["predict"]
        args = {}
        if self.ai_system.metric_manager.user_config is not None \
                and "bias" in self.ai_system.metric_manager.user_config \
                and "args" in self.ai_system.metric_manager.user_config["bias"]:
            args = self.ai_system.metric_manager.user_config["bias"]["args"]

        self.metrics["explained_variance"].value = sklearn.metrics.explained_variance_score(data.y, preds, **args.get("explained_variance", {}))
        self.metrics["mean_absolute_error"].value = sklearn.metrics.mean_absolute_error(data.y, preds, **args.get("mean_absolute_error", {}))
        self.metrics["mean_absolute_percentage_error"].value = sklearn.metrics.mean_absolute_percentage_error(
            data.y, preds, **args.get("mean_absolute_percentage_error", {})
        )
        self.metrics["mean_gamma_deviance"].value = sklearn.metrics.mean_gamma_deviance(data.y, preds, **args.get("mean_gamma_deviance", {}))
        self.metrics["mean_poisson_deviance"].value = sklearn.metrics.mean_poisson_deviance(data.y, preds, **args.get("mean_poisson_deviance", {}))
        self.metrics["mean_squared_error"].value = sklearn.metrics.mean_squared_error(data.y, preds, **args.get("mean_squared_error", {}))
        self.metrics["mean_squared_log_error"].value = sklearn.metrics.mean_squared_log_error(data.y, preds, **args.get("mean_squared_log_error", {}))
        self.metrics["median_absolute_error"].value = sklearn.metrics.median_absolute_error(data.y, preds, **args.get("median_absolute_error", {}))
        self.metrics["r2"].value = sklearn.metrics.r2_score(data.y, preds, **args.get("r2", {}))
