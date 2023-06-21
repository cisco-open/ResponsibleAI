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
from torchmetrics.image.inception import InceptionScore
import torch
import numpy as np


class ImageGenerationInception(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        images = data_dict["generate_image"]
        inception_score = {"mean": 0, "std": 0}
        inception_score["mean"], inception_score["std"] = _inception(images)
        self.metrics["inception"].value = inception_score


def _inception(images):
    shape = list(images.shape)
    shape = shape[-3:]
    shape.insert(0, -1)
    images_shaped = images.reshape(tuple(shape))
    images_shaped = _convert_float_to_uint8(images_shaped)
    torch_img = torch.from_numpy(images_shaped)
    inception = InceptionScore()
    inception.update(torch_img)
    result = inception.compute()
    return result[0].item(), result[1].item()


def _convert_float_to_uint8(images):
    if np.issubdtype(images.dtype, np.floating):
        images = (images * 255).astype(np.uint8)
    return images
