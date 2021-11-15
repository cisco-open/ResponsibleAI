from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *

import pandas as pd

__all__ = ['compatibility']

compatibility = {"type_restriction": "classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "name" : "group_fairness",
    "compatibility" : {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness", "Equal Treatment"],
    "complexity_class": "linear",
    "metrics": {
        "disparate_impact_ratio": {
            "display_name": "Disparate Impact",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "statistical_parity_difference": {
            "display_name": "Statistical Parity Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": "Disparate Impact describes how preferential the treatment is for privileged groups when compared to unprivileged groups."
        },
        "equal_opportunity_difference": {
            "display_name": "Equal Opportunity Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "average_odds_difference": {
            "display_name": "Average Odds Difference",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "average_odds_error": {
            "display_name": "Average Odds Error",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        }
    }
}


class GroupFairnessMetricGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            priv_group = None
            if self.ai_system.user_config is not None and "equal_treatment" in self.ai_system.user_config and "priv_group" in self.ai_system.user_config["equal_treatment"]:
                priv_group = self.ai_system.user_config["equal_treatment"]["priv_group"]

            y = _convert_to_ai360(self, data)
            self.metrics['disparate_impact_ratio'].value = _disparate_impact_ratio(y, preds, prot_attr=priv_group[0], pos_label=priv_group[1])
            self.metrics['statistical_parity_difference'].value = _statistical_parity_difference(y, preds, prot_attr=priv_group[0], pos_label=priv_group[1])
            self.metrics['equal_opportunity_difference'].value = _equal_opportunity_difference(y, preds, prot_attr=priv_group[0], pos_label=priv_group[1])
            self.metrics['average_odds_difference'].value = _average_odds_difference(y, preds, prot_attr=priv_group[0], pos_label=priv_group[1])
            self.metrics['average_odds_error'].value = _average_odds_error(y, preds, prot_attr=priv_group[0], pos_label=priv_group[1])



def _convert_to_ai360(metric_group, data):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df = pd.DataFrame(data.X, columns=names)
    df['y'] = data.y
    X, y = standardize_dataset(df, prot_attr=['race'], target='y')
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
