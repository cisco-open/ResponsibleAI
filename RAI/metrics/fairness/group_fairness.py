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
import pandas as pd
import os
from RAI.metrics.ai360_helper import get_classification_dataset
from aif360.sklearn.metrics import average_odds_error


class GroupFairnessMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    @classmethod
    def is_compatible(cls, ai_system):
        compatible = super().is_compatible(ai_system)
        return compatible \
            and "fairness" in ai_system.metric_manager.user_config \
            and "protected_attributes" in ai_system.metric_manager.user_config["fairness"] \
            and len(ai_system.metric_manager.user_config["fairness"]["protected_attributes"]) > 0 \
            and "positive_label" in ai_system.metric_manager.user_config["fairness"]

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        data = data_dict["data"]
        preds = data_dict["predict"]
        priv_group_list = []
        unpriv_group_list = []
        prot_attr = []
        if self.ai_system.metric_manager.user_config is not None and "fairness" in self.ai_system.metric_manager.user_config and "priv_group" in \
                self.ai_system.metric_manager.user_config["fairness"]:
            prot_attr = self.ai_system.metric_manager.user_config["fairness"]["protected_attributes"]
            for group in self.ai_system.metric_manager.user_config["fairness"]["priv_group"]:
                priv_group_list.append(
                    {group: self.ai_system.metric_manager.user_config["fairness"]["priv_group"][group]["privileged"]})
                unpriv_group_list.append(
                    {group: self.ai_system.metric_manager.user_config["fairness"]["priv_group"][group]["unprivileged"]})

        cd = get_classification_dataset(self, data, preds, prot_attr, priv_group_list, unpriv_group_list)
        self.metrics['disparate_impact_ratio'].value = cd.disparate_impact()
        self.metrics['statistical_parity_difference'].value = cd.statistical_parity_difference()
        self.metrics['equal_opportunity_difference'].value = cd.equal_opportunity_difference()
        self.metrics['average_odds_difference'].value = cd.average_odds_difference()
        self.metrics['average_odds_error'].value = self._average_odds_error(data, preds, prot_attr)
        self.metrics['between_group_generalized_entropy_error'].value = cd.between_group_generalized_entropy_index()

    def _average_odds_error(self, data, preds, prot_attr):
        names = [feature.name for feature in self.ai_system.meta_database.features if feature.categorical]
        df = pd.DataFrame(data.categorical, columns=names)
        df['y'] = data.y
        gt_series = df['y'].squeeze()
        gt_series.index = df['race']
        return average_odds_error(gt_series, preds, prot_attr=prot_attr)
