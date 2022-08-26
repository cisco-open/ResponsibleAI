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