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

        gen_images = gen_images.reshape((-1, gen_images.shape[2], gen_images.shape[3], gen_images.shape[4]))
        gt_images = gt_images.reshape((-1, gt_images.shape[2], gt_images.shape[3], gt_images.shape[4]))

        gt_images = torch.from_numpy(_convert_float_to_uint8(gt_images))
        gen_images = torch.from_numpy(_convert_float_to_uint8(gen_images))

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