from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import pandas as pd
from RAI.metrics.AIF360.datasets import BinaryLabelDataset
from RAI.metrics.AIF360.metrics import ClassificationMetric
from RAI.utils import compare_runtimes

__all__ = ['compatibility']

compatibility = {"type_restriction": "binary_classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name": "prediction_fairness",
    "display_name" : "Classification Fairness Metrics",
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
            "range": [0, 1],
            "explanation": "Average of absolute difference in FPR and TPR for unprivileged and privileged groups:"
        },
        "between-all-groups-coefficient-of-variation": {
            "display_name": "BG Coefficient of Variation",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The between-group coefficient of variation is two times the square root of the entropy index between all groups."
        },
        "between-all-groups-generalized-entropy-index": {
            "display_name": "All Groups Gen Entropy Index",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Entropy index for all groups with protected attributes."
        },
        "between-all-groups-theil-index": {
            "display_name": "Between all Groups Theil Index",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Equivalent to the between group generalized entropy index with a=1"
        },
        "between-group-coefficient-of-variation": {
            "display_name": "BG Coefficient of Var",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": "Two times the square root of the between group generalized entropy index with α=2."
        },
        "between-group-generalized-entropy-index": {
            "display_name": "BG General Entropy Index",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Entropy index between the privileged and unprivileged groups."
        },
        "between-group-theil-index": {
            "display_name": "BG Theil Index",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Equals tthe between group generalized entropy index with α=1"
        },
        "coefficient-of-variation": {
            "display_name": "Coefficient of Variation",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Equals two times the square root of the generalized entropy index with α=2."
        },
        "consistency": {
            "display_name": "Consistency",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": "Indicates the similarity of labels for similiar instances."
        },
        "differential-fairness-bias-amplification": {
            "display_name": "Diff Fairness Bias Amplification",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": "Bias amplification is the difference in smoothed EDF between the classifier and the original dataset. Positive values mean the bias increased due to the classifier."
        },
        "error-rate": {
            "display_name": "Error Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The percentage of predictions that were incorrect."
        },
        "error-rate-difference": {
            "display_name": "Error Rate Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Compares error rate between an unprivileged group and a privileged group."
        },
        "error-rate-ratio": {
            "display_name": "Error Rate Ratio",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The ratio of error rate between privileged and unprivileged groups."
        },
        "false-discovery-rate": {
            "display_name": "False Discovery Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The percentage of positive predictions that were false positives."
        },
        "false-discovery-rate-difference": {
            "display_name": "False Discovery Rate Diff",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "The difference in false discovery rate between unprivileged and privileged groups."
        },
        "false-discovery-rate-ratio": {
            "display_name": "False Discovery Rate Ratio",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The ratio of false discovery rates between unprivileged and privileged groups."
        },
        "false-negative-rate": {
            "display_name": "False Negative Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The probability of falsely predicting a false outcome."
        },
        "false-negative-rate-difference": {
            "display_name": "False Negative Rate Diff",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": "The difference in false negative rate between unprivileged and privileged groups."
        },
        "false-negative-rate-ratio": {
            "display_name": "False Negative Rate Ratio",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The ratio of false negative rate between unprivileged and privileged groups."
        },
        "generalized-entropy-index": {
            "display_name": "Generalized Entropy Index",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": "Unified individual and group fairness metric. From T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar, “A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018"
        },
        "generalized-true-negative-rate": {
            "display_name": "Gen True Negative Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of generalized true negatives divided by the number of negatives."
        },
        "generalized-true-positive-rate": {
            "display_name": "Gen True Positive Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of generalized true positives divided by the number of positives."
        },
        "negative-predictive-value": {
            "display_name": "Negative Predictive Value",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The chance that negative prediction is correct."
        },
        "num-false-negatives": {
            "display_name": "False Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of false negatives."
        },
        "num-false-positives": {
            "display_name": "False Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of false positives"
        },
        "num-generalized-false-negatives": {
            "display_name": "Gen False Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Weighted sum of 1-predicted scores where true labels are favorable."
        },
        "num-generalized-false-positives": {
            "display_name": "Gen False Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Weighted sum of predicted scores where true labels are unfavorable."
        },
        "num-generalized-true-negatives": {
            "display_name": "Gen True Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Weighted sum of 1 - predicted scores where true labels are unfavorable."
        },
        "num-generalized-true-positives": {
            "display_name": "Generalized True Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Weighted sum of predicted scores where true labels are favorable."
        },
        "num-instances": {
            "display_name": "Instance Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of examples seen."
        },
        "num-negatives": {
            "display_name": "Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of negative examples seen."
        },
        "num-positives": {
            "display_name": "Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of positive examples seen."
        },
        "num-pred-negatives": {
            "display_name": "Negative Prediction Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of negative predictions seen."
        },
        "num-pred-positives": {
            "display_name": "Positive Prediction Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of positive predictions seen."
        },
        "num-true-negatives": {
            "display_name": "True Negative Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of times the classifier correctly predicted an example to be negative."
        },
        "num-true-positives": {
            "display_name": "True Positive Count",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "The number of times the classifier correctly predicted an example to be positive."
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
            "has_range": True,
            "range": [0, 1],
            "explanation": "Fairness metric from: Foulds, James R., et al. 'An intersectional definition of fairness.' 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020."
        },
        "true-negative-rate": {
            "display_name": "True Negative Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The number of true negatives divided by the total number of negatives."
        },
        "true-positive-rate": {
            "display_name": "True Positive Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The number of true positives divided by the total number of positives."
        },
        "true-positive-rate-difference": {
            "display_name": "True Positive Rate Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "The difference in true positive rate between unprivileged and privileged groups."
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
                     and "fairness" in ai_system.metric_manager.user_config \
                     and "protected_attributes" in ai_system.metric_manager.user_config["fairness"] \
                     and  len(ai_system.metric_manager.user_config["fairness"]["protected_attributes"])>0 \
                     and "priv_group" in ai_system.metric_manager.user_config["fairness"] \
                     and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"), _config["complexity_class"])
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
            if self.ai_system.metric_manager.user_config is not None and "fairness" in self.ai_system.metric_manager.user_config and "priv_group" in \
                    self.ai_system.metric_manager.user_config["fairness"]:
                prot_attr = self.ai_system.metric_manager.user_config["fairness"]["protected_attributes"]
                for group in self.ai_system.metric_manager.user_config["fairness"]["priv_group"]:
                    priv_group_list.append({group: self.ai_system.metric_manager.user_config["fairness"]["priv_group"][group]["privileged"]})
                    unpriv_group_list.append({group: self.ai_system.metric_manager.user_config["fairness"]["priv_group"][group]["unprivileged"]})

            cd = get_class_dataset(self, data, preds, prot_attr, priv_group_list, unpriv_group_list)
            self.metrics['average-abs-odds-difference'].value = cd.average_odds_difference()
            self.metrics['between-all-groups-coefficient-of-variation'].value = cd.between_all_groups_coefficient_of_variation()
            self.metrics['between-all-groups-generalized-entropy-index'].value = cd.between_all_groups_generalized_entropy_index()
            self.metrics['between-all-groups-theil-index'].value = cd.between_all_groups_theil_index()
            self.metrics['between-group-coefficient-of-variation'].value = cd.between_group_coefficient_of_variation()
            self.metrics['between-group-generalized-entropy-index'].value = cd.between_group_generalized_entropy_index()
            self.metrics['between-group-theil-index'].value = cd.between_group_theil_index()
            self.metrics['coefficient-of-variation'].value = cd.coefficient_of_variation()
            self.metrics['consistency'].value = cd.consistency()[0]
            self.metrics['differential-fairness-bias-amplification'].value = cd.differential_fairness_bias_amplification()
            self.metrics['error-rate'].value = cd.error_rate()
            self.metrics['error-rate-difference'].value = cd.error_rate_difference()
            self.metrics['error-rate-ratio'].value = cd.error_rate_ratio()
            self.metrics['false-discovery-rate'].value = cd.false_discovery_rate()
            self.metrics['false-discovery-rate-difference'].value = cd.false_discovery_rate_difference()
            self.metrics['false-discovery-rate-ratio'].value = cd.false_discovery_rate_ratio()
            self.metrics['false-negative-rate'].value = cd.false_negative_rate()
            self.metrics['false-negative-rate-difference'].value = cd.false_negative_rate_difference()
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
