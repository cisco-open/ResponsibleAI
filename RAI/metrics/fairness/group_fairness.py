from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import numpy as np
import pandas as pd
import os


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

