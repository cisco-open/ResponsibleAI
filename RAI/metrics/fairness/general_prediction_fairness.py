from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import pandas as pd
from RAI.metrics.AIF360.datasets import BinaryLabelDataset
from RAI.metrics.AIF360.metrics import ClassificationMetric

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name": "prediction_fairness",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness", "General Fairness"],
    "complexity_class": "linear",
    "metrics": {
        "average-abs-odds-difference": {
            "display_name": "Average Abs Odds Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "between-all-groups-coefficient-of-variation": {
            "display_name": "BG Coefficient of Variation",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "between-all-groups-generalized-entropy-index": {
            "display_name": "All Groups Gen Entropy Index",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "between-all-groups-theil-index": {
            "display_name": "Between all Groups Theil Index",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "between-group-coefficient-of-variation": {
            "display_name": "BG Coefficient of Var",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "between-group-generalized-entropy-index": {
            "display_name": "BG General Entropy Index",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "between-group-theil-index": {
            "display_name": "BG Theil Index",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "coefficient-of-variation": {
            "display_name": "Coefficient of Variation",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "consistency": {
            "display_name": "Consistency",
            "type": "vector",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "differential-fairness-bias-amplification": {
            "display_name": "Diff Fairness Bias Amplification",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "error-rate": {
            "display_name": "Error Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "error-rate-difference": {
            "display_name": "Error Rate Difference",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "error-rate-ratio": {
            "display_name": "Error Rate Ratio",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "false-discovery-rate": {
            "display_name": "False Discovery Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "false-discovery-rate-difference": {
            "display_name": "False Discovery Rate Diff",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "false-discovery-rate-ratio": {
            "display_name": "False Discovery Rate Ratio",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "false-negative-rate": {
            "display_name": "False Negative Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "false-negative-rate-difference": {
            "display_name": "False Negative Rate Diff",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "false-negative-rate-ratio": {
            "display_name": "False Negative Rate Ratio",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "generalized-entropy-index": {
            "display_name": "Generalized Entropy Index",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "generalized-true-negative-rate": {
            "display_name": "Gen True Negative Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "generalized-true-positive-rate": {
            "display_name": "Gen True Positive Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "negative-predictive-value": {
            "display_name": "Negative Predictive Value",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-false-negatives": {
            "display_name": "False Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-false-positives": {
            "display_name": "False Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-generalized-false-negatives": {
            "display_name": "Gen False Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-generalized-false-positives": {
            "display_name": "Gen False Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-generalized-true-negatives": {
            "display_name": "Gen True Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-generalized-true-positives": {
            "display_name": "Generalized True Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-instances": {
            "display_name": "Instance Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-negatives": {
            "display_name": "Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-positives": {
            "display_name": "Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-pred-negatives": {
            "display_name": "Negative Prediction Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-pred-positives": {
            "display_name": "Positive Prediction Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-true-negatives": {
            "display_name": "True Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "num-true-positives": {
            "display_name": "True Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "positive-predictive-value": {
            "display_name": "Positive Predictive Value",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "smoothed-empirical-differential-fairness": {
            "display_name": "Smoothed Emp Diff Fairness",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "true-negative-rate": {
            "display_name": "True Negative Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "true-positive-rate": {
            "display_name": "True Positive Rate",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },
        "true-positive-rate-difference": {
            "display_name": "True Positive Rate Difference",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": ""
        },


    }
}


class GeneralPredictionFairnessGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible \
                     and "fairness" in ai_system.user_config \
                     and "protected_attributes" in ai_system.user_config["fairness"] \
                     and "priv_group" in ai_system.user_config["fairness"]
        return compatible

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            priv_group_list = []
            unpriv_group_list = []
            prot_attr = []
            if self.ai_system.user_config is not None and "fairness" in self.ai_system.user_config and "priv_group" in \
                    self.ai_system.user_config["fairness"]:
                prot_attr = self.ai_system.user_config["fairness"]["protected_attributes"]
                for group in self.ai_system.user_config["fairness"]["priv_group"]:
                    priv_group_list.append({group: self.ai_system.user_config["fairness"]["priv_group"][group]["privileged"]})
                    unpriv_group_list.append({group: self.ai_system.user_config["fairness"]["priv_group"][group]["unprivileged"]})

            cd = get_class_dataset(self, data, preds, prot_attr, priv_group_list, unpriv_group_list)
            self.metrics['average-abs-odds-difference'].value = cd.average_odds_difference()
            self.metrics['between-all-groups-coefficient-of-variation'].value = cd.between_all_groups_coefficient_of_variation()
            self.metrics['between-all-groups-generalized-entropy-index'].value = cd.between_all_groups_generalized_entropy_index()
            self.metrics['between-all-groups-theil-index'].value = cd.between_all_groups_theil_index()
            self.metrics['between-group-coefficient-of-variation'].value = cd.between_group_coefficient_of_variation()
            self.metrics['between-group-generalized-entropy-index'].value = cd.between_group_generalized_entropy_index()
            self.metrics['between-group-theil-index'].value = cd.between_group_theil_index()
            self.metrics['coefficient-of-variation'].value = cd.coefficient_of_variation()
            self.metrics['consistency'].value = cd.consistency()
            self.metrics['differential-fairness-bias-amplification'].value = cd.differential_fairness_bias_amplification()
            self.metrics['error-rate'].value = cd.error_rate()
            self.metrics['error-rate-difference'].value = cd.error_rate_difference()
            self.metrics['error-rate-ratio'].value = cd.error_rate_ratio()
            self.metrics['false-discovery-rate'].value = cd.false_discovery_rate()
            self.metrics['false-discovery-rate-difference'].value = cd.false_discovery_rate_difference()
            self.metrics['false-discovery-rate-ratio'].value = cd.false_discovery_rate_ratio()
            self.metrics['false-negative-rate'].value = cd.false_negative_rate()
            self.metrics['false-negative-rate-difference'].value = cd.false_discovery_rate_difference()
            self.metrics['false-negative-rate-ratio'].value = cd.false_negative_rate_ratio()
            self.metrics['generalized-entropy-index'].value = cd.generalized_entropy_index()
            self.metrics['generalized-true-negative-rate'].value = cd.generalized_true_negative_rate()
            self.metrics['generalized-true-positive-rate'].value = cd.generalized_true_positive_rate()
            self.metrics['negative-predictive-value'].value = cd.negative_predictive_value()
            self.metrics['num-false-negatives'].value = cd.num_false_negatives()
            self.metrics['num-false-positives'].value = cd.num_false_positives()
            self.metrics['num-generalized-false-negatives'].value = cd.num_generalized_false_negatives()
            self.metrics['num-generalized-false-positives'].value = cd.num_generalized_false_positives()
            self.metrics['num-generalized-true-negatives'].value = cd.num_generalized_true_negatives()
            self.metrics['num-generalized-true-positives'].value = cd.num_generalized_true_positives()
            self.metrics['num-instances'].value = cd.num_instances()
            self.metrics['num-negatives'].value = cd.num_negatives()
            self.metrics['num-positives'].value = cd.num_positives()
            self.metrics['num-pred-negatives'].value = cd.num_pred_negatives()
            self.metrics['num-pred-positives'].value = cd.num_pred_positives()
            self.metrics['num-true-negatives'].value = cd.num_true_negatives()
            self.metrics['num-true-positives'].value = cd.num_true_positives()
            self.metrics['positive-predictive-value'].value = cd.positive_predictive_value()
            self.metrics['smoothed-empirical-differential-fairness'].value = cd.smoothed_empirical_differential_fairness()
            self.metrics['true-negative-rate'].value = cd.true_negative_rate()
            self.metrics['true-positive-rate'].value = cd.true_positive_rate()
            self.metrics['true-positive-rate-difference'].value = cd.true_positive_rate_difference()


def get_class_dataset(metric_group, data, preds, prot_attr, priv_group_list, unpriv_group_list):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df1 = pd.DataFrame(data.X, columns=names)
    df1['y'] = data.y
    df2 = pd.DataFrame(data.X, columns=names)
    df2['y'] = preds
    binDataset1 = BinaryLabelDataset(df=df1, label_names=['y'], protected_attribute_names=prot_attr)
    binDataset2 = BinaryLabelDataset(df=df2, label_names=['y'], protected_attribute_names=prot_attr)
    return ClassificationMetric(binDataset1, binDataset2, unprivileged_groups=unpriv_group_list, privileged_groups=priv_group_list)
