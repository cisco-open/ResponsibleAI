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
from RAI.metrics.ai360_helper import get_classification_dataset
import os


class GeneralPredictionFairnessGroup(MetricGroup, class_location=os.path.abspath(__file__)):
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
            and "priv_group" in ai_system.metric_manager.user_config["fairness"]

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
                priv_group_list.append({group: self.ai_system.metric_manager.user_config["fairness"]["priv_group"][group]["privileged"]})
                unpriv_group_list.append({group: self.ai_system.metric_manager.user_config["fairness"]["priv_group"][group]["unprivileged"]})

        cd = get_classification_dataset(self, data, preds, prot_attr, priv_group_list, unpriv_group_list)
        self.metrics['average_odds_difference'].value = cd.average_odds_difference()
        self.metrics['between_all_groups_coefficient_of_variation'].value = cd.between_all_groups_coefficient_of_variation()
        self.metrics['between_all_groups_generalized_entropy_index'].value = cd.between_all_groups_generalized_entropy_index()
        self.metrics['between_all_groups_theil_index'].value = cd.between_all_groups_theil_index()
        self.metrics['between_group_coefficient_of_variation'].value = cd.between_group_coefficient_of_variation()
        self.metrics['between_group_generalized_entropy_index'].value = cd.between_group_generalized_entropy_index()
        self.metrics['between_group_theil_index'].value = cd.between_group_theil_index()
        self.metrics['coefficient_of_variation'].value = cd.coefficient_of_variation()
        self.metrics['consistency'].value = cd.consistency()[0]
        self.metrics['differential_fairness_bias_amplification'].value = cd.differential_fairness_bias_amplification()
        self.metrics['error_rate'].value = cd.error_rate()
        self.metrics['error_rate_difference'].value = cd.error_rate_difference()
        self.metrics['error_rate_ratio'].value = cd.error_rate_ratio()
        self.metrics['false_discovery_rate'].value = cd.false_discovery_rate()
        self.metrics['false_discovery_rate_difference'].value = cd.false_discovery_rate_difference()
        self.metrics['false_discovery_rate_ratio'].value = cd.false_discovery_rate_ratio()
        self.metrics['false_negative_rate'].value = cd.false_negative_rate()
        self.metrics['false_negative_rate_difference'].value = cd.false_negative_rate_difference()
        self.metrics['false_negative_rate_ratio'].value = cd.false_negative_rate_ratio()
        self.metrics['generalized_entropy_index'].value = cd.generalized_entropy_index()
        self.metrics['generalized_true_negative_rate'].value = cd.generalized_true_negative_rate()
        self.metrics['generalized_true_positive_rate'].value = cd.generalized_true_positive_rate()
        self.metrics['negative_predictive_value'].value = cd.negative_predictive_value()
        self.metrics['num_false_negatives'].value = cd.num_false_negatives()
        self.metrics['num_false_positives'].value = cd.num_false_positives()
        self.metrics['num_generalized_false_negatives'].value = cd.num_generalized_false_negatives()
        self.metrics['num_generalized_false_positives'].value = cd.num_generalized_false_positives()
        self.metrics['num_generalized_true_negatives'].value = cd.num_generalized_true_negatives()
        self.metrics['num_generalized_true_positives'].value = cd.num_generalized_true_positives()
        self.metrics['num_instances'].value = cd.num_instances()
        self.metrics['num_negatives'].value = cd.num_negatives()
        self.metrics['num_positives'].value = cd.num_positives()
        self.metrics['num_pred_negatives'].value = cd.num_pred_negatives()
        self.metrics['num_pred_positives'].value = cd.num_pred_positives()
        self.metrics['num_true_negatives'].value = cd.num_true_negatives()
        self.metrics['num_true_positives'].value = cd.num_true_positives()
        self.metrics['positive_predictive_value'].value = cd.positive_predictive_value()
        self.metrics['smoothed_empirical_differential_fairness'].value = cd.smoothed_empirical_differential_fairness()
        self.metrics['true_negative_rate'].value = cd.true_negative_rate()
        self.metrics['true_positive_rate'].value = cd.true_positive_rate()
        self.metrics['true_positive_rate_difference'].value = cd.true_positive_rate_difference()
