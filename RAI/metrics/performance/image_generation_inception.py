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
        images = np.array(images)
        inception_score = {"mean": 0, "std": 0}
        inception_score["mean"], inception_score["std"] = _inception(images)
        self.metrics["inception"].value = inception_score


def _inception(images):
    images = images.reshape((-1, images.shape[2], images.shape[3], images.shape[4]))
    images = _convert_float_to_uint8(images)
    torch_img = torch.from_numpy(images)
    inception = InceptionScore()
    inception.update(torch_img)
    result = inception.compute()
    return result[0].item(), result[1].item()


def _convert_float_to_uint8(images):
    if np.issubdtype(images.dtype, np.floating):
        images = (images * 255).astype(np.uint8)
    return images