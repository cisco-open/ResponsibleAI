from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import sklearn

__all__ = ['compatibility']
 
# Log loss, roc and brier score have been removed. s

_config = {
    "name" : "performance_cl",
    "compatibility" : {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "stats",
    "dependency_list": [],
    "tags": ["performance", "Classification"],
    "complexity_class": "linear",
    "metrics": {
        "accuracy": {
            "display_name": "Accuracy",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Accuracy describes how well a model performed at classifying data."
        },
        "auc": {
            "display_name": "Area Under the Curve",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Describes the ability of a classifier to distinguish between classes."
        },
        "balanced_accuracy": {
            "display_name": "Balanced Accuracy",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Balanced Accuracy describes how well a performed by taking the average recall across each class."
        },
        "confusion_matrix": {
            "display_name": "Confusion Matrix",
            "type": "matrix",
            "tags": [],
            "has_range": False,
            "range": None,
            "explanation": "The Confusion Matrix summarizes performance and can highlight areas of weakness where incorrect classification is common."
        },
        "f1": {
            "display_name": "F1 Score",
            "type": "vector",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The F1 score is a weighted average between a models precision and recall scores",
        },
        "f1_avg": {
            "display_name": "F1 Score",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "The F1 score is a weighted average between a models precision and recall scores",
        },

        "fp_rate_avg": {
            "display_name": "Average False Positive Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "FP Rate describes what percentage of wrong predictions were false positives.",
        },
        "fp_rate": {
            "display_name": "False Positive Rate",
            "type": "vector",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "FP Rate describes what percentage of wrong predictions were false positives.",
        },
        
        "jaccard_score": {
            "display_name": "Jaccard Score",
            "type": "vector",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Jaccard Score measures the similarity of two two sets of data, and returns a result from 0 to 100%.",
        },
        "jaccard_score_avg": {
            "display_name": "Jaccard Score",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Jaccard Score measures the similarity of two two sets of data, and returns a result from 0 to 100%.",
        },
        
        "precision_score": {
            "display_name": "Precision Score",
            "type": "vector",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Precision Scores indicates a models ability to not label a positive sample as negative.",
        },
        "precision_score_avg": {
            "display_name": "Precision Score",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Precision Scores indicates a models ability to not label a positive sample as negative.",
        },

        "recall_score": {
            "display_name": "Recall Score",
            "type": "vector",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Recall Scores indicates a models ability to classify all positive image samples",
        },
        "recall_score_avg": {
            "display_name": "Recall Score",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Recall Scores indicates a models ability to classify all positive image samples",
        },

        "tp_rate": {
            "display_name": "True Positive Rate",
            "type": "vector",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "True Positive Rate is the probability that a positive example will be predicted to be positive.",
        },
        "tp_rate_avg": {
            "display_name": "True Positive Rate",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "True Positive Rate is the probability that a positive example will be predicted to be positive.",
        },
    }
}


class PerformanceClassificationMetricGroup(MetricGroup, config=_config):
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
            if self.ai_system.user_config is not None and "bias" in self.ai_system.user_config and "args" in self.ai_system.user_config["bias"]:
                args = self.ai_system.user_config["bias"]["args"]

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

            self.metrics["tp_rate"].value = _tp_rate(fptn, **args.get("tp_rate", {}))
            self.metrics["tp_rate_avg"].value = np.mean(self.metrics["tp_rate"].value)

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

