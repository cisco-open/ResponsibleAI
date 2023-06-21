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
from RAI.all_types import all_output_requirements
from RAI.dataset import NumpyData
import datetime
import os


class MetadataGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def compute(self, data_dict):
        self.metrics["date"].value = self._get_time()
        self.metrics["description"].value = self.ai_system.model.description

        samples = 0
        if "data" in data_dict and data_dict["data"] is not None:
            data = data_dict["data"]
            if isinstance(data, NumpyData):
                if data.X is not None:
                    samples = data.X.shape[0]
                elif data.y is not None:
                    samples = len(data.y)
        else:
            for output_type in all_output_requirements:
                if output_type in data_dict and data_dict[output_type] is not None:
                    samples = len(data_dict[output_type])

        self.metrics["sample_count"].value = samples
        self.metrics["task_type"].value = self.ai_system.task
        if self.ai_system.model.agent:
            self.metrics["model"].value = str(self.ai_system.model.agent)
        else:
            self.metrics["model"].value = "None"

        self.metrics["tag"].value = data_dict["tag"]

    def compute_batch(self, data_dict):
        prev_samples = self.metrics["sample_count"].value
        prev_samples = 0 if prev_samples is None else prev_samples
        self.compute(data_dict)
        self.metrics["sample_count"].value += prev_samples

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) \
            + "-" + "{:02d}".format(now.month) \
            + "-" + "{:02d}".format(now.day) \
            + " " + "{:02d}".format(now.hour) \
            + ":" + "{:02d}".format(now.minute) \
            + ":" + "{:02d}".format(now.second)
