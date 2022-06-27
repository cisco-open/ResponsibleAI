from RAI.metrics.metric_group import MetricGroup
import numpy as np
import scipy.stats
import warnings
import os
from RAI.utils.utils import calculate_per_all_features

class StatMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        args = {}
        if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
            args = self.ai_system.metric_manager.user_config["stats"]["args"]
        data = data_dict["data"]
        scalar_data = data.scalar

        self.metrics["mean"].value = np.mean(scalar_data, **args.get("mean", {}), axis=0)
        self.metrics["covariance"].value = np.cov(scalar_data.T, **args.get("covariance", {}))
        self.metrics["num_nan_rows"].value = np.count_nonzero(np.isnan(data.X).any(axis=1))
        self.metrics["percent_nan_rows"].value = self.metrics["num_nan_rows"].value/np.shape(np.asarray(data.X))[0]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.metrics["geometric_mean"].value = scipy.stats.mstats.gmean(scalar_data)

        self.metrics["mode"].value = scipy.stats.mstats.mode(scalar_data)[0][0]
        self.metrics["skew"].value = scipy.stats.mstats.skew(scalar_data)
        self.metrics["variation"].value = scipy.stats.mstats.variation(scalar_data)
        self.metrics["sem"].value = scipy.stats.mstats.sem(scalar_data)
        self.metrics['kurtosis'].value = scipy.stats.mstats.kurtosis(scalar_data)

        features = self.ai_system.meta_database.features
        map = self.ai_system.meta_database.scalar_map

        self.metrics["frozen_mean_mean"].value = [None]*len(features)
        self.metrics["frozen_mean_variance"].value = [None]*len(features)
        self.metrics["frozen_mean_std"].value = [None]*len(features)
        self.metrics["frozen_variance_mean"].value = [None]*len(features)
        self.metrics["frozen_variance_variance"].value = [None]*len(features)
        self.metrics["frozen_variance_std"].value = [None]*len(features)
        self.metrics["frozen_std_mean"].value = [None]*len(features)
        self.metrics["frozen_std_variance"].value = [None]*len(features)
        self.metrics["frozen_std_std"].value = [None]*len(features)

        values = calculate_per_all_features(scipy.stats.mvsdist, map, features, data.scalar)

        for i, value in enumerate(values):
            if value is not None:
                self.metrics["frozen_mean_mean"].value[i] = value[0].mean()
                self.metrics["frozen_mean_variance"].value[i] = value[0].var()
                self.metrics["frozen_mean_std"].value[i] = value[0].std()
                self.metrics["frozen_variance_mean"].value[i] = value[1].mean()
                self.metrics["frozen_variance_variance"].value[i] = value[1].var()
                self.metrics["frozen_variance_std"].value[i] = value[1].std()
                self.metrics["frozen_std_mean"].value[i] = value[2].mean()
                self.metrics["frozen_std_variance"].value[i] = value[2].var()
                self.metrics["frozen_std_std"].value[i] = value[2].std()

        self.metrics["kstat_1"].value = calculate_per_all_features(scipy.stats.kstat, map, features, data.scalar, 1)
        self.metrics["kstat_2"].value = calculate_per_all_features(scipy.stats.kstat, map, features, data.scalar, 2)
        self.metrics["kstat_3"].value = calculate_per_all_features(scipy.stats.kstat, map, features, data.scalar, 3)
        self.metrics["kstat_4"].value = calculate_per_all_features(scipy.stats.kstat, map, features, data.scalar, 4)
        self.metrics["kstatvar"].value = calculate_per_all_features(scipy.stats.kstatvar, map, features, data.scalar)
        self.metrics["iqr"].value = calculate_per_all_features(scipy.stats.iqr, map, features, data.scalar)

        self.metrics["bayes_mean"].value = [None] * len(features)
        self.metrics["bayes_mean_avg"].value = [None] * len(features)
        self.metrics["bayes_variance"].value = [None] * len(features)
        self.metrics["bayes_variance_avg"].value = [None] * len(features)
        self.metrics["bayes_std"].value = [None] * len(features)
        self.metrics["bayes_std_avg"].value = [None] * len(features)
        values = calculate_per_all_features(scipy.stats.bayes_mvs, map, features, data.scalar)

        for i, value in enumerate(values):
            if value is not None:
                self.metrics["bayes_mean"].value[i] = value[0][1]
                self.metrics["bayes_mean_avg"].value[i] = value[0][0]
                self.metrics["bayes_variance"].value[i] = value[1][1]
                self.metrics["bayes_variance_avg"].value[i] = value[1][0]
                self.metrics["bayes_std"].value[i] = value[2][1]
                self.metrics["bayes_std_avg"].value[i] = value[2][0]
