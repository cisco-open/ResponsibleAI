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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torch
import numpy as np


class ImageGeneration(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        self.max_samples = 500

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        gt_images = data_dict["data"].y
        gen_images = data_dict["generate_image"]
        gt_images = gt_images[:self.max_samples]
        gen_images = gen_images[:self.max_samples]

        gen_images = np.array(gen_images)
        gt_images = np.array(gt_images)

        img_shape = list(gen_images.shape)
        img_shape = img_shape[-3:]
        img_shape.insert(0, -1)

        gen_shaped_images = gen_images.reshape(tuple(img_shape))
        gt_shaped_images = gt_images.reshape(tuple(img_shape))

        gt_images = torch.from_numpy(_convert_float_to_uint8(gt_shaped_images))
        gen_images = torch.from_numpy(_convert_float_to_uint8(gen_shaped_images))

        # I lack the memory to run this!
        self.metrics["fid"].value = _fid(gt_images, gen_images)
        # https://torchmetrics.readthedocs.io/en/stable/image/kernel_inception_distance.html
        self.metrics["kid"].value = _kid(gt_images, gen_images)


def _kid(gt_images, gen_images):
    kid = KernelInceptionDistance(subset_size=50)
    kid.update(gt_images, real=True)
    kid.update(gen_images, real=False)
    kid_mean, kid_std = kid.compute()
    return {"mean": kid_mean.item(), "std": kid_std.item()}


def _fid(gt_images, gen_images):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(gt_images, real=True)
    fid.update(gen_images, real=False)
    return fid.compute().item()


def _convert_float_to_uint8(images):
    if np.issubdtype(images.dtype, np.floating):
        images = (images * 255).astype(np.uint8)
    return images
