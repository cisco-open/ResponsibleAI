from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import sklearn

__all__ = ['compatibility']

compatibility = {"type_restriction": "classification", "output_restriction": "choice"}

# Log loss, roc and brier score have been removed. s

_config = {
    "src": "stats",
    "dependency_list": [],
    "tags": ["bias"],
    "complexity_class": "linear",
    "metrics": {
        "accuracy": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1]

        },
        "balanced_accuracy": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1]
        },
        "confusion_matrix": {
            "type": "other",
            "tags": [],
            "has_range": False,
        },
        "f1": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "fp_rate": {
            "type": "other",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "jaccard_score": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "precision_score": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "recall_score": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
        "tp_rate": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
        },
    }
}


class ClassBiasMetricGroup(MetricGroup, name="class_bias"):
    def __init__(self, ai_system, config=_config) -> None:
        super().__init__(ai_system, config)
        self.ai_system = ai_system
        self.config = config
        self.compatibility = {"type_restriction": "classification", "output_restriction": "choice"}

    def is_compatible(self, ai_system):
        return self.compatibility["type_restriction"] is None or self.compatibility["type_restriction"] == ai_system.task.type

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            args = {}
            if "bias" in self.ai_system.user_config and "args" in self.ai_system.user_config["bias"]:
                args = self.ai_system.user_config["bias"]["args"]

            self.metrics["accuracy"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("accuracy", {}))
            self.metrics["balanced_accuracy"].value = sklearn.metrics.balanced_accuracy_score(data.y, preds, **args.get("balanced_accuracy", {}))
            self.metrics["confusion_matrix"].value = sklearn.metrics.confusion_matrix(data.y, preds, **args.get("confusion_matrix", {}))
            fptn = get_fptn(self.metrics["confusion_matrix"].value)  # TP, TN, FP, FN values. Used quite a bit.
            self.metrics["fp_rate"].value = _fp_rate(fptn, **args.get("fp_rate", {}))
            self.metrics["f1"].value = sklearn.metrics.f1_score(data.y, preds, average=None, **args.get("f1", {}))
            self.metrics["jaccard_score"].value = sklearn.metrics.jaccard_score(data.y, preds, average=None, **args.get("jaccard_score", {}))
            self.metrics["precision_score"].value = _precision_score(fptn, **args.get("precision_score", {}))
            self.metrics["recall_score"].value = _recall_score(fptn, **args.get("recall_score", {}))
            self.metrics["tp_rate"].value = _tp_rate(fptn, **args.get("tp_rate", {}))


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

