from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import scipy.stats

# Move config to external .json? 
_config = {
    "name": "summary_stats",
    "compatibility" : {"type_restriction": None, "output_restriction": None},
    "dependency_list": [],
    "tags": ["stats", "Summary Stats"],
    "complexity_class": "linear",
    "metrics": {
        "mean": {
            "display_name": "Mean",
            "type": "vector",
            "has_range": False,
            "range": None,
            "explanation": "Mean is the expected value of data.",
        },
        "covariance": {
            "display_name": "Covariance Matrix",
            "type": "matrix",
            "has_range": False,
            "range": None,
            "explanation": "A Covariance Matrix shows the directional relationship between different data members.",
        },
        "num-Nan-rows": {
            "display_name": "Number of NaN Rows",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Num Nan Rows indicates the number of NaN rows found in the data.",
        },
        "percent-Nan-rows": {
            "display_name": "Percentage of NaN Rows",
            "type": "numeric",
            "has_range": True,
            "range": [0, 1],
            "explanation": "Percent Nan Rows indicates the percentage of rows that are NaN in the data.",
        },
        "geometric-mean": {
            "display_name": "Geometric Mean",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "kurtosis": {
            "display_name": "Kurtosis",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "mode": {
            "display_name": "Mode",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "skew": {
            "display_name": "Skew",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "kstat-1": {
            "display_name": "K-Statistic 1",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "kstat-2": {
            "display_name": "K-Statistic 2",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "kstat-3": {
            "display_name": "K-Statistic 3",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "kstat-4": {
            "display_name": "K-Statistic 4",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "kstatvar": {
            "display_name": "K-Statistic Variance",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "variance": {
            "display_name": "Variance",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "iqr": {
            "display_name": "Interquartile Range",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "sem": {
            "display_name": "Standard Error of the Mean",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "bayes-mean": {
            "display_name": "Mean Confidence Interval",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "bayes-mean-avg": {
            "display_name": "Mean Confidence Interval Average",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "bayes-var": {
            "display_name": "Variance Confidence Interval",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "bayes-var-avg": {
            "display_name": "Variance Confidence Interval Avg",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "bayes-std": {
            "display_name": "Standard Deviation Confidence Interval",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "bayes-std-avg": {
            "display_name": "Standard Deviation Confidence Interval Average",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "frozen-mean-mean": {
            "display_name": "Frozen Mean Mean",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "frozen-mean-std": {
            "display_name": "Frozen Mean Standard Deviation",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "frozen-variance-mean": {
            "display_name": "Frozen Variance Mean",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "frozen-variance-std": {
            "display_name": "Frozen Variance Standard Deviation",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "frozen-std-mean": {
            "display_name": "Frozen Standard Deviation Mean",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
        "frozen-std-std": {
            "display_name": "Frozen Stdev Standard Deviation",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "",
        },
    }
}

# Type (Regression, Classification, Data | probability, numeric)


class StatMetricGroup(MetricGroup, config=_config):
    compatibility = {"type_restriction": None, "output_restriction": None}

    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.user_config is not None and "stats" in self.ai_system.user_config and "args" in self.ai_system.user_config["stats"]:
                args = self.ai_system.user_config["stats"]["args"]

            data = data_dict["data"]
            self.metrics["mean"].value = np.mean(data.X, **args.get("mean", {}), axis=0)
            self.metrics["covariance"].value = np.cov(data.X.T, **args.get("covariance", {}))
            self.metrics["num-Nan-rows"].value = np.count_nonzero(np.isnan(data.X).any(axis=1))
            self.metrics["percent-Nan-rows"].value = self.metrics["num-Nan-rows"].value/np.shape(np.asarray(data.X))[0]
            self.metrics["geometric-mean"].value = scipy.stats.gmean(data.X)
            self.metrics["mode"].value = scipy.stats.mode(data.X)[0]
            self.metrics["skew"].value = scipy.stats.skew(data.X)
            self.metrics["kstat-1"].value = scipy.stats.kstat(data.X, 1)
            self.metrics["kstat-2"].value = scipy.stats.kstat(data.X, 2)
            self.metrics["kstat-3"].value = scipy.stats.kstat(data.X, 3)
            self.metrics["kstat-4"].value = scipy.stats.kstat(data.X, 4)
            self.metrics["kstatvar"].value = scipy.stats.kstatvar(data.X)
            self.metrics["variance"].value = scipy.stats.variation(data.X)
            self.metrics["iqr"].value = scipy.stats.iqr(data.X)
            self.metrics["sem"].value = scipy.stats.sem(data.X)
            bMean, bVar, bStd = scipy.stats.bayes_mvs(data.X)
            self.metrics["bayes-mean"].value = bMean[1]
            self.metrics["bayes-mean-avg"].value = bMean[0]
            self.metrics["bayes-var"].value = bVar[1]
            self.metrics["bayes-var-avg"].value = bVar[0]
            self.metrics["bayes-std"].value = bStd[1]
            self.metrics["bayes-std-avg"].value = bStd[0]
            fMean, fVar, fStd = scipy.stats.mvsdist(data.X)
            self.metrics["frozen-mean-mean"].value = fMean.mean()
            self.metrics["frozen-mean-std"].value = fMean.std()
            self.metrics["frozen-variance-mean"].value = fVar.mean()
            self.metrics["frozen-variance-std"].value = fVar.std()
            self.metrics["frozen-std-mean"].value = fStd.mean()
            self.metrics["frozen-std-std"].value = fStd.std()

