from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import numpy as np
import pandas as pd
from RAI.utils import compare_runtimes

__all__ = ['compatibility']

compatibility = {"type_restriction": "classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name" : "group_fairness",
    "display_name" : "Group Fairness Metrics",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness", "Group Fairness"],
    "complexity_class": "linear",
    "metrics": {
        "disparate_impact_ratio": {
            "display_name": "Disparate Impact",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Compares the proportion of individuals that receive a positive outcome between two groups."
        },
        "statistical_parity_difference": {
            "display_name": "Statistical Parity Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Finds the difference in positive outcomes seen between a privileged and unprivileged group groups."
        },
        "between_group_generalized_entropy_error": {
            "display_name": "BG Generalized Entropy Error",
            "type": "numeric",
            "tags": [],
            "has_range": False,
            "range": [None, None],
            "explanation": "Fairness metric from T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar, “A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018."
        },
        "equal_opportunity_difference": {
            "display_name": "Equal Opportunity Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Indicates difference in recall scores between unprivileged groups and privileged groups. Values close to 0 are better."
        },
        "average_odds_difference": {
            "display_name": "Average Odds Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [-1, 1],
            "explanation": "Returns the average difference in False positive rate and true positive rate between privileged and unprivileged groups."
        },
        "average_odds_error": {
            "display_name": "Average Odds Error",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Indicates the average of the absolute difference in FPR and TPR for the unprivileged and privileged groups"
        }
    }
}


class GroupFairnessMetricGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible and "fairness" in ai_system.metric_manager.user_config \
                     and "protected_attributes" in ai_system.metric_manager.user_config["fairness"] \
                     and  len(ai_system.metric_manager.user_config["fairness"]["protected_attributes"])>0 \
                     and "positive_label" in ai_system.metric_manager.user_config["fairness"] \
                     and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"), _config["complexity_class"])
        return compatible

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            prot_attr = self.ai_system.metric_manager.user_config["fairness"]["protected_attributes"]
            pos_label = self.ai_system.metric_manager.user_config["fairness"]["positive_label"]

            y = _convert_to_ai360(self, data, prot_attr)
            self.metrics['disparate_impact_ratio'].value = _disparate_impact_ratio(y, preds, prot_attr=prot_attr[0], pos_label=pos_label)
            self.metrics['statistical_parity_difference'].value = _statistical_parity_difference(y, preds, prot_attr=prot_attr[0], pos_label=pos_label)
            self.metrics['equal_opportunity_difference'].value = _equal_opportunity_difference(y, preds, prot_attr=prot_attr[0], pos_label=pos_label)
            self.metrics['average_odds_difference'].value = _average_odds_difference(y, preds, prot_attr=prot_attr[0], pos_label=pos_label)
            self.metrics['average_odds_error'].value = _average_odds_error(y, preds, prot_attr=prot_attr[0], pos_label=pos_label)
            self.metrics['between_group_generalized_entropy_error'].value = _between_group_generalized_entropy_error(y, preds, prot_attr=prot_attr[0], pos_label=pos_label)


def _convert_to_ai360(metric_group, data, prot_attr):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df = pd.DataFrame(data.X, columns=names)
    df['y'] = data.y
    X, y = standardize_dataset(df, prot_attr=prot_attr, target='y')
    return y


def _disparate_impact_ratio(*y, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None):
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return ratio(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                 pos_label=pos_label, sample_weight=sample_weight)


def _statistical_parity_difference(*y, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None):
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return difference(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                      pos_label=pos_label, sample_weight=sample_weight)


def _equal_opportunity_difference(y_true, y_pred, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None):
    return difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)


def _average_odds_difference(y_true, y_pred, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None):
    fpr_diff = -difference(specificity_score, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           pos_label=pos_label, sample_weight=sample_weight)
    tpr_diff = difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (tpr_diff + fpr_diff) / 2


def _average_odds_error(y_true, y_pred, prot_attr=None, pos_label=1, sample_weight=None):
    priv_group = check_groups(y_true, prot_attr=prot_attr)[0][0]
    fpr_diff = -difference(specificity_score, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           pos_label=pos_label, sample_weight=sample_weight)
    tpr_diff = difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (abs(tpr_diff) + abs(fpr_diff)) / 2


def _between_group_generalized_entropy_error(y_true, y_pred, prot_attr=None, priv_group=None, alpha=2, pos_label=1):
    groups, _ = check_groups(y_true, prot_attr)
    b = np.empty_like(y_true, dtype='float')
    if priv_group is not None:
        groups = [1 if g == priv_group else 0 for g in groups]
    for g in np.unique(groups):
        b[groups == g] = (1 + (y_pred[groups == g] == pos_label)
                            - (y_true[groups == g] == pos_label)).mean()
    return _generalized_entropy_index(b, alpha=alpha)


def _generalized_entropy_index(b, alpha=2):
    if alpha == 0:
        return -(np.log(b / b.mean()) / b.mean()).mean()
    elif alpha == 1:
        # moving the b inside the log allows for 0 values
        return (np.log((b / b.mean())**b) / b.mean()).mean()
    else:
        return ((b / b.mean())**alpha - 1).mean() / (alpha * (alpha - 1))


def statistical_parity_difference(*y, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None):
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return difference(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                      pos_label=pos_label, sample_weight=sample_weight)

