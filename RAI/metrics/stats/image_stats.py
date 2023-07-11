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
import os
from RAI.utils import convert_float32_to_float64
import numpy as np


class ImageStatsGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        self._batch_total_examples = 0
        self._batch_average = [0, 0, 0]
        self._batch_d_squared = [0.0, 0.0, 0.0]

    def update(self, data):
        pass

    def compute(self, data_dict):
        data = data_dict["data"]
        # images are of shape [examples, image columns, c, w, h]
        images = data.image
        means = convert_float32_to_float64(images.mean(axis=(0, 1, 3, 4)))  # images.mean((0, 1, 3, 4))
        self.metrics["mean"].value = {"red": convert_float32_to_float64(means[0]),
                                      "green": convert_float32_to_float64(means[1]),
                                      "blue": convert_float32_to_float64(means[2])}
        std = convert_float32_to_float64(images.std(axis=(0, 1, 3, 4)))
        self.metrics["std"].value = {"red": convert_float32_to_float64(std[0]),
                                     "green": convert_float32_to_float64(std[1]),
                                     "blue": convert_float32_to_float64(std[2])}  # images.std((0, 1, 3, 4))

    def reset(self):
        super().reset()
        self._batch_total_examples = 0
        self._batch_average = [0, 0, 0]
        self._batch_d_squared = [0.0, 0.0, 0.0]

    def compute_batch(self, data_dict):
        data = data_dict["data"]
        images = data.image
        for image in images:
            self._batch_total_examples += 1
            means = convert_float32_to_float64(image.mean(axis=(0, 2, 3)))
            for i in range(len(self._batch_average)):
                prev_avg = self._batch_average[i]
                self._batch_average[i] = (self._batch_average[i] - means[i]) / self._batch_total_examples
                self._batch_d_squared[i] = self._batch_d_squared[i] + (means[i] - self._batch_average[i]) * (means[i] - prev_avg)

        for i in range(len(self._batch_d_squared)):
            self._batch_d_squared[i] = self._batch_d_squared[i] / self._batch_total_examples
            self._batch_d_squared[i] = np.sqrt(self._batch_d_squared[i])
        self.metrics["mean"].value = {"red": self._batch_average[0], "green": self._batch_average[1], "blue": self._batch_average[2]}
        self.metrics["std"].value = {"red": self._batch_d_squared[0], "green": self._batch_d_squared[1], "blue": self._batch_d_squared[2]}
        pass
