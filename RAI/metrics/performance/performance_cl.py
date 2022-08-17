from RAI.metrics.metric_group import MetricGroup
from RAI.dataset import NumpyData
import numpy as np
import sklearn
import os
import warnings


class PerformanceClassificationMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        self._y_data = []
        self._preds = []
        
    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        data = data_dict["data"]
        preds = data_dict["predict"]
        args = {}
        y_data = data.y

        warnings.filterwarnings("ignore")
        self.metrics["accuracy"].value = sklearn.metrics.accuracy_score(y_data, preds, **args.get("accuracy", {}))
        self.metrics["balanced_accuracy"].value = sklearn.metrics.balanced_accuracy_score(y_data, preds, **args.get("balanced_accuracy", {}))
        self.metrics["confusion_matrix"].value = sklearn.metrics.confusion_matrix(y_data, preds, **args.get("confusion_matrix", {}))
        fptn = get_fptn(self.metrics["confusion_matrix"].value)  # TP, TN, FP, FN values. Used quite a bit.

        self.metrics["fp_rate"].value = _fp_rate(fptn, **args.get("fp_rate", {}))
        self.metrics["fp_rate_avg"].value = np.mean(self.metrics["fp_rate"].value)

        self.metrics["f1"].value = sklearn.metrics.f1_score(y_data, preds, average=None, **args.get("f1", {}))
        self.metrics["f1_avg"].value = np.mean(self.metrics["f1"].value)

        self.metrics["jaccard_score"].value = sklearn.metrics.jaccard_score(y_data, preds, average=None, **args.get("jaccard_score", {}))
        self.metrics["jaccard_score_avg"].value = np.mean(self.metrics["jaccard_score"].value)

        self.metrics["precision_score"].value = _precision_score(fptn, **args.get("precision_score", {}))
        self.metrics["precision_score_avg"].value = np.mean(self.metrics["precision_score"].value)

        self.metrics["recall_score"].value = _recall_score(fptn, **args.get("recall_score", {}))
        self.metrics["recall_score_avg"].value = np.mean(self.metrics["recall_score"].value)

    def reset(self):
        super().reset()
        self._y_data = []
        self._preds = []

    def compute_batch(self, data_dict):
        data = data_dict["data"]
        preds = data_dict["predict"]
        y_data = data.y

        # classification uses integer based y values, assume it can be held in memory for now
        # TODO: replace with stream based calculations in future
        self._y_data.extend(y_data)
        self._preds.extend(preds)

    def finalize_batch_compute(self):
        temp_dict = {}
        resulting_data = NumpyData(None, self._y_data)
        temp_dict["data"] = resulting_data
        temp_dict["predict"] = self._preds
        self.compute(temp_dict)

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

