from RAI.metrics.metric_group import MetricGroup
import os
from RAI.utils import convert_float32_to_float64
import numpy as np


class ImageStatsGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        images = data_dict["data"].image

        # images are of shape [examples, image columns, c, w, h]
        means = convert_float32_to_float64(images.mean(axis=(0, 1, 3, 4)))  # images.mean((0, 1, 3, 4))
        self.metrics["mean"].value = {"red": convert_float32_to_float64([0]),
                                      "green": convert_float32_to_float64(means[1]),
                                      "blue": convert_float32_to_float64(means[2])}
        std = convert_float32_to_float64(images.std(axis=(0, 1, 3, 4)))
        self.metrics["std"].value = {"red": convert_float32_to_float64(std[0]),
                                     "green": convert_float32_to_float64(std[1]),
                                     "blue": convert_float32_to_float64(std[2])}  # images.std((0, 1, 3, 4))

