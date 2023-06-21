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
import numpy as np
import scipy.stats
import warnings
import os
import pandas as pd
from RAI.utils.utils import calculate_per_mapped_features, map_to_feature_dict, map_to_feature_array, convert_float32_to_float64


class StatMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def compute(self, data_dict):
        args = {}
        if self.ai_system.metric_manager.user_config is not None \
                and "stats" in self.ai_system.metric_manager.user_config \
                and "args" in self.ai_system.metric_manager.user_config["stats"]:
            args = self.ai_system.metric_manager.user_config["stats"]["args"]
        data = data_dict["data"]
        scalar_data = data.scalar
        scalar_map = self.ai_system.meta_database.scalar_map
        features = self.ai_system.meta_database.features

        self.metrics["mean"].value = map_to_feature_dict(
            np.mean(scalar_data, **args.get("mean", {}), axis=0), features, scalar_map
        )
        self.metrics["mean"].value = convert_float32_to_float64(self.metrics["mean"].value)
        self.metrics["covariance"].value = map_to_feature_array(
            np.cov(scalar_data.T, **args.get("covariance", {})), features, scalar_map
        )
        self.metrics["num_nan_rows"].value = np.count_nonzero(pd.isna(data.X).any(axis=1))
        self.metrics["percent_nan_rows"].value = self.metrics["num_nan_rows"].value / np.shape(np.asarray(data.X))[0]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                self.metrics["geometric_mean"].value = map_to_feature_dict(
                    scipy.stats.mstats.gmean(scalar_data), features, scalar_map
                )
                self.metrics['geometric_mean'].value = convert_float32_to_float64(self.metrics['geometric_mean'].value)
            except Exception:
                self.metrics["geometric_mean"].value = None

        self.metrics["mode"].value = map_to_feature_dict(
            scipy.stats.mstats.mode(scalar_data)[0][0], features, scalar_map
        )
        self.metrics["skew"].value = map_to_feature_dict(
            scipy.stats.mstats.skew(scalar_data), features, scalar_map)
        self.metrics['skew'].value = convert_float32_to_float64(self.metrics['skew'].value)
        self.metrics["variation"].value = map_to_feature_dict(
            scipy.stats.mstats.variation(scalar_data), features, scalar_map
        )
        self.metrics['variation'].value = convert_float32_to_float64(self.metrics['variation'].value)

        self.metrics["median"].value = map_to_feature_dict(np.median(scalar_data, axis=0), features, scalar_map)
        self.metrics["quantile_1"].value = map_to_feature_dict(
            np.quantile(scalar_data, 0.25, axis=0), features, scalar_map
        )
        self.metrics["quantile_3"].value = map_to_feature_dict(
            np.quantile(scalar_data, 0.75, axis=0), features, scalar_map
        )
        self.metrics["min"].value = map_to_feature_dict(np.min(scalar_data, axis=0), features, scalar_map)
        self.metrics["max"].value = map_to_feature_dict(np.max(scalar_data, axis=0), features, scalar_map)
        self.metrics["standard_deviation"].value = map_to_feature_dict(
            np.std(scalar_data, axis=0), features, scalar_map
        )

        self.metrics["sem"].value = map_to_feature_dict(scipy.stats.mstats.sem(scalar_data), features, scalar_map)
        self.metrics['kurtosis'].value = map_to_feature_dict(
            scipy.stats.mstats.kurtosis(scalar_data), features, scalar_map
        )
        self.metrics['kurtosis'].value = convert_float32_to_float64(self.metrics['kurtosis'].value)

        features = self.ai_system.meta_database.features
        map = self.ai_system.meta_database.scalar_map

        self.metrics["frozen_mean_mean"].value = {}
        self.metrics["frozen_mean_variance"].value = {}
        self.metrics["frozen_mean_std"].value = {}
        self.metrics["frozen_variance_mean"].value = {}
        self.metrics["frozen_variance_variance"].value = {}
        self.metrics["frozen_variance_std"].value = {}
        self.metrics["frozen_std_mean"].value = {}
        self.metrics["frozen_std_variance"].value = {}
        self.metrics["frozen_std_std"].value = {}

        values = calculate_per_mapped_features(scipy.stats.mvsdist, map, features, data.scalar, to_array=False)

        for key in values:
            if values[key] is not None:
                self.metrics["frozen_mean_mean"].value[key] = values[key][0].mean()
                self.metrics["frozen_mean_variance"].value[key] = values[key][0].var()
                self.metrics["frozen_mean_std"].value[key] = values[key][0].std()
                self.metrics["frozen_variance_mean"].value[key] = values[key][1].mean()
                self.metrics["frozen_variance_variance"].value[key] = values[key][1].var()
                self.metrics["frozen_variance_std"].value[key] = values[key][1].std()
                self.metrics["frozen_std_mean"].value[key] = values[key][2].mean()
                self.metrics["frozen_std_variance"].value[key] = values[key][2].var()
                self.metrics["frozen_std_std"].value[key] = values[key][2].std()

        self.metrics["kstat_1"].value = calculate_per_mapped_features(
            scipy.stats.kstat, map, features, data.scalar, 1, to_array=False
        )
        self.metrics["kstat_2"].value = calculate_per_mapped_features(
            scipy.stats.kstat, map, features, data.scalar, 2, to_array=False
        )
        self.metrics["kstat_3"].value = calculate_per_mapped_features(
            scipy.stats.kstat, map, features, data.scalar, 3, to_array=False
        )
        self.metrics["kstat_4"].value = calculate_per_mapped_features(
            scipy.stats.kstat, map, features, data.scalar, 4, to_array=False
        )
        self.metrics["kstatvar"].value = calculate_per_mapped_features(
            scipy.stats.kstatvar, map, features, data.scalar, to_array=False
        )
        self.metrics["iqr"].value = calculate_per_mapped_features(
            scipy.stats.iqr, map, features, data.scalar, to_array=False
        )

        self.metrics["bayes_mean"].value = {}
        self.metrics["bayes_mean_avg"].value = {}
        self.metrics["bayes_variance"].value = {}
        self.metrics["bayes_variance_avg"].value = {}
        self.metrics["bayes_std"].value = {}
        self.metrics["bayes_std_avg"].value = {}
        values = calculate_per_mapped_features(scipy.stats.bayes_mvs, map, features, data.scalar, to_array=False)

        for key in values:
            if values[key] is not None:
                self.metrics["bayes_mean"].value[key] = values[key][0][1]
                self.metrics["bayes_mean_avg"].value[key] = values[key][0][0]
                self.metrics["bayes_variance"].value[key] = values[key][1][1]
                self.metrics["bayes_variance_avg"].value[key] = values[key][1][0]
                self.metrics["bayes_std"].value[key] = values[key][2][1]
                self.metrics["bayes_std_avg"].value[key] = values[key][2][0]
