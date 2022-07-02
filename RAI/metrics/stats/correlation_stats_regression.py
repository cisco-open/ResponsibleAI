from RAI.metrics.metric_group import MetricGroup
import scipy.stats
from RAI.utils.utils import calculate_per_mapped_features
import os


class CorrelationStatRegression(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        args = {}
        if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
            args = self.ai_system.metric_manager.user_config["stats"]["args"]
        data = data_dict["data"]
        map = self.ai_system.meta_database.scalar_map
        features = self.ai_system.meta_database.features

        self.metrics["pearson_correlation"].value = calculate_per_mapped_features(scipy.stats.pearsonr, map, features, data.scalar, data.y)
        self.metrics["spearman_correlation"].value = calculate_per_mapped_features(scipy.stats.spearmanr, map, features, data.scalar, data.y)
