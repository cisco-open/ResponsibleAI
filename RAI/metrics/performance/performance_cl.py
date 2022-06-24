from RAI.metrics.metric_group import MetricGroup
import numpy as np
import sklearn
import os


class PerformanceClassificationMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
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
            args = {}
            if self.ai_system.metric_manager.user_config is not None and "bias" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["bias"]:
                args = self.ai_system.metric_manager.user_config["bias"]["args"]

            self.metrics["accuracy"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("accuracy", {}))
            self.metrics["balanced_accuracy"].value = sklearn.metrics.balanced_accuracy_score(data.y, preds, **args.get("balanced_accuracy", {}))
            self.metrics["confusion_matrix"].value = sklearn.metrics.confusion_matrix(data.y, preds, **args.get("confusion_matrix", {}))
            fptn = get_fptn(self.metrics["confusion_matrix"].value)  # TP, TN, FP, FN values. Used quite a bit.
            
            self.metrics["fp_rate"].value = _fp_rate(fptn, **args.get("fp_rate", {}))
            self.metrics["fp_rate_avg"].value = np.mean(self.metrics["fp_rate"].value)
            
            self.metrics["f1"].value = sklearn.metrics.f1_score(data.y, preds, average=None, **args.get("f1", {}))
            self.metrics["f1_avg"].value = np.mean(self.metrics["f1"].value)
            
            self.metrics["jaccard_score"].value = sklearn.metrics.jaccard_score(data.y, preds, average=None, **args.get("jaccard_score", {}))
            self.metrics["jaccard_score_avg"].value = np.mean(self.metrics["jaccard_score"].value)
            
            self.metrics["precision_score"].value = _precision_score(fptn, **args.get("precision_score", {}))
            self.metrics["precision_score_avg"].value = np.mean(self.metrics["precision_score"].value)
            
            self.metrics["recall_score"].value = _recall_score(fptn, **args.get("recall_score", {}))
            self.metrics["recall_score_avg"].value = np.mean(self.metrics["recall_score"].value )

            # Revisit this, should roc_curve be calculated per each label?
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(data.y, preds, pos_label=None)
            self.metrics["auc"].value = sklearn.metrics.auc(fpr, tpr)



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

