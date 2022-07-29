from RAI.metrics.metric_group import MetricGroup
import os
import numpy as np
from RAI.utils import map_to_feature_array, map_to_feature_dict, convert_float32_to_float64


class ImageStatsGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        images = data_dict["data"].image
        images = np.array(images)

        # images are of shape [examples, image columns, c, w, h]
        # TODO: needs to be calculated per image column, use calculate_per_feature
        # TODO: Add per channel calculation, using commented out code
        self.metrics["mean"].value = convert_float32_to_float64(images.mean())  # images.mean((0, 1, 3, 4))
        self.metrics["std"].value = convert_float32_to_float64(images.std())  # images.std((0, 1, 3, 4))

