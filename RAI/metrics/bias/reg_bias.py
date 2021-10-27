from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import sklearn


# Log loss, roc and brier score have been removed.

_config = {
    "src": "stats",
    "dependency_list": [],
    "tags": ["bias"],
    "complexity_class": "linear",
    "metrics": {
        "explained_variance": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1]
        },
        "mean_absolute_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1]
        },
        "mean_absolute_percentage_error": {
            "type": "other",
            "tags": [],
            "has_range": False,
        },
        "mean_gamma_deviance": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "mean_poisson_deviance": {
            "type": "other",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "mean_squared_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "mean_squared_log_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "median_absolute_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "r2": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
    }
}


class RegBiasMetricGroup(MetricGroup, name="reg_bias"):
    def __init__(self, ai_system, config=_config) -> None:
        super().__init__(ai_system, config)
        self._bias_config = None

    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]

def get_fptn(confusion_matrix):
    result = {'fp': confusion_matrix.sum(axis=0) - np.diag(confusion_matrix),
              'fn': confusion_matrix.sum(axis=1) - np.diag(confusion_matrix),
              'tp': np.diag(confusion_matrix)}
    result['tn'] = confusion_matrix.sum() - result['fp'] - result['fn'] - result['tp']
    return result


def _fp_rate(fptn):
        return fptn['fp'] / (fptn['fp'] + fptn['tn'])


def _tp_rate(fptn):
    return fptn['tp'] / (fptn['tp'] + fptn['fn'])


def _precision_score(fptn):
    return fptn['tp'] / (fptn['tp'] + fptn['fp'])


def _recall_score(fptn):
    return fptn['tp'] / (fptn['tp'] + fptn['fn'])

