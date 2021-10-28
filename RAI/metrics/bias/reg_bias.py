from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import sklearn


_config = {
    "src": "stats",
    "category": "bias",
    "dependency_list": [],
    "tags": ["bias"],
    "complexity_class": "linear",
    "metrics": {
        "explained_variance": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Measures discrepancy between a model and actual data, higher values are better",
        },
        "mean_absolute_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Indicates the average residual error, lower is better.",
        },
        "mean_absolute_percentage_error": {
            "type": "other",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Indicates the how inaccurate predictions were from actual values, lower is better.",
        },
        "mean_gamma_deviance": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculated by taking the Tweedie Deviance with a power of 2.",
        },
        "mean_poisson_deviance": {
            "type": "other",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Calculated by taking the Tweedie Deviance with a power of 1.",
        },
        "mean_squared_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Mean Squared Error indicates the the performance of a model, lower is better.",
        },
        "mean_squared_log_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Mean Squared Log Error indicates the average squared logarithmic residual error, lower is better.",
        },
        "median_absolute_error": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Median Absolute Error indicates the median value for absolute residual error, lower is better.",
        },
        "r2": {
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Indicates how well a model fits the data, higher is better.",
        },
    }
}


class RegBiasMetricGroup(MetricGroup, name="reg_bias"):
    def __init__(self, ai_system, config=_config) -> None:
        super().__init__(ai_system, config)
        self._ai_system = ai_system
        self.config = config
        self.compatibility = {"type_restriction": "regression", "output_restriction": None}

    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            args = {}
            if "bias" in self.ai_system.user_config and "args" in self.ai_system.user_config["bias"]:
                args = self.ai_system.user_config["bias"]["args"]

            self.metrics["explained_variance"].value = sklearn.metrics.explained_variance_score(data.y, preds, **args.get("explained_variance", {}))
            self.metrics["mean_absolute_error"].value = sklearn.metrics.mean_absolute_error(data.y, preds, **args.get("mean_absolute_error", {}))
            self.metrics["mean_absolute_percentage_error"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("accuracy", {}))
            self.metrics["mean_gamma_deviance"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("mean_gamma_deviance", {}))
            self.metrics["mean_poisson_deviance"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("mean_poisson_deviance", {}))
            self.metrics["mean_squared_error"].value = sklearn.metrics.mean_squared_error(data.y, preds, **args.get("mean_squared_error", {}))
            self.metrics["mean_squared_log_error"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("mean_squared_log_error", {}))
            self.metrics["median_absolute_error"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("median_absolute_error", {}))
            self.metrics["r2"].value = sklearn.metrics.accuracy_score(data.y, preds, **args.get("r2", {}))
