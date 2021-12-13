from RAI.metrics.metric_group import MetricGroup
import math
import numpy as np
import scipy.stats
import warnings
            
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
            "explanation": "The Geometric Mean shows the central tendency of a set of numbers. It is calculated by taking the n-th root of the product of n numbers.",
        },
        "kurtosis": {
            "display_name": "Kurtosis",
            "type": "vector",
            "has_range": None,
            "range": [None, None],
            "explanation": "Kurtosis is the tailedness of skewdness of data. ",
        },
        "mode": {
            "display_name": "Mode",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "Mode is the most often appearing value in a set of data values.",
        },
        "skew": {
            "display_name": "Skew",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "Skew measures the symmetry of a data set.",
        },
        "kstat-1": {
            "display_name": "K-Statistic 1",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "A unique, symmetric and unbiased estimator of the first cumulant kappa-n",
        },
        "kstat-2": {
            "display_name": "K-Statistic 2",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "A unique, symmetric and unbiased estimator of the second cumulant kappa-n",
        },
        "kstat-3": {
            "display_name": "K-Statistic 3",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "A unique, symmetric and unbiased estimator of the third cumulant kappa-n",
        },
        "kstat-4": {
            "display_name": "K-Statistic 4",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "A unique, symmetric and unbiased estimator of the fourth cumulant kappa-n",
        },
        "kstatvar": {
            "display_name": "K-Statistic Variance",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "",
        },
        "variation": {
            "display_name": "Coefficient of Variation",
            "type": "vector",
            "has_range": True,
            "range": [0, None],
            "explanation": "Indicates the ratio of the standard deviation to the mean. Higher values indicate higher spread.",
        },
        "iqr": {
            "display_name": "Interquartile Range",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "IQR is the difference between the 75th and 25th percentile of data. A robust measure of dispersion.",
        },
        "sem": {
            "display_name": "Standard Error of the Mean",
            "type": "vector",
            "has_range": True,
            "range": [0, None],
            "explanation": "Standard Error of the mean is the standard deviation of the sampling distribution of the mean. Calculated by dividing the standard deviation by the root of the sample size.",
        },
        "bayes-mean": {
            "display_name": "Mean Confidence Interval",
            "type": "vector",
            "has_range": False,
            "range": [None, None],
            "explanation": "Bayesian estimate of upper and lower bounds of the population mean",
        },
        "bayes-mean-avg": {
            "display_name": "Mean Confidence Interval Average",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "Bayesian estimate of the population mean",
        },
        "bayes-var": {
            "display_name": "Variance Confidence Interval",
            "type": "vector",
            "has_range": True,
            "range": [0, None],
            "explanation": "Bayesian estimate of upper and lower bounds of the population mean",
        },
        "bayes-var-avg": {
            "display_name": "Variance Confidence Interval Avg",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Bayesian estimate of the population mean",
        },
        "bayes-std": {
            "display_name": "Stdev Confidence Interval",
            "type": "vector",
            "has_range": True,
            "range": [0, None],
            "explanation": "Bayesian estimate of upper and lower bounds of the population standard deviation",
        },
        "bayes-std-avg": {
            "display_name": "Stdev Confidence Interval Avg",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Bayesian estimate of the population standard deviation.",
        },
        "frozen-mean-mean": {
            "display_name": "Frozen Mean Mean",
            "type": "numeric",
            "has_range": False,
            "range": [None, None],
            "explanation": "Population mean estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-mean-var": {
            "display_name": "Frozen Mean Variance",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population variance of the mean estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-mean-std": {
            "display_name": "Frozen Mean Standard Deviation",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population standard deviation of the mean estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-variance-mean": {
            "display_name": "Frozen Variance Mean",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population variance of the mean estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-variance-var": {
            "display_name": "Frozen Variance Variance",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population variance of the variance estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-variance-std": {
            "display_name": "Frozen Variance Standard Deviation",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population variance of the standard deviation estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-std-mean": {
            "display_name": "Frozen Standard Deviation Mean",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population mean of the standard deviation estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-std-var": {
            "display_name": "Frozen Stdev Variance",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population variance of the standard deviation estimator. https://scholarsarchive.byu.edu/facpub/278/",
        },
        "frozen-std-std": {
            "display_name": "Frozen Stdev Standard Deviation",
            "type": "numeric",
            "has_range": True,
            "range": [0, None],
            "explanation": "Population Standard deviation of the standard deviation estimator. https://scholarsarchive.byu.edu/facpub/278/",
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

            mask = np.zeros_like(data.X)
            mask = mask + self.ai_system.meta_database.scalar_mask
            masked_data = np.ma.masked_array(data.X, mask)

            self.metrics["mean"].value = np.mean(masked_data, **args.get("mean", {}), axis=0)
            self.metrics["covariance"].value = np.cov(masked_data.T, **args.get("covariance", {}))
            self.metrics["num-Nan-rows"].value = np.count_nonzero(np.isnan(data.X).any(axis=1))
            self.metrics["percent-Nan-rows"].value = self.metrics["num-Nan-rows"].value/np.shape(np.asarray(data.X))[0]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.metrics["geometric-mean"].value = scipy.stats.mstats.gmean(masked_data)
            
            self.metrics["mode"].value = scipy.stats.mstats.mode(data.X)[0]
            self.metrics["skew"].value = scipy.stats.mstats.skew(masked_data)
            self.metrics["variation"].value = scipy.stats.mstats.variation(masked_data)
            self.metrics["sem"].value = scipy.stats.mstats.sem(masked_data)
            self.metrics['kurtosis'].value = scipy.stats.mstats.kurtosis(masked_data)

            scalar_data = _get_scalar_data(data.X, self.ai_system.meta_database.scalar_mask)

            # Singular Valued
            fMean, fVar, fStd = scipy.stats.mvsdist(scalar_data)
            self.metrics["frozen-mean-mean"].value = fMean.mean()
            self.metrics["frozen-mean-var"].value = fMean.var()
            self.metrics["frozen-mean-std"].value = fMean.std()
            self.metrics["frozen-variance-mean"].value = fVar.mean()
            self.metrics["frozen-variance-var"].value = fVar.var()
            self.metrics["frozen-variance-std"].value = fVar.std()
            self.metrics["frozen-std-mean"].value = fStd.mean()
            self.metrics["frozen-std-var"].value = fStd.var()
            self.metrics["frozen-std-std"].value = fStd.std()

            # Singular Valued
            self.metrics["kstat-1"].value = scipy.stats.kstat(scalar_data, 1)
            self.metrics["kstat-2"].value = scipy.stats.kstat(scalar_data, 2)
            self.metrics["kstat-3"].value = scipy.stats.kstat(scalar_data, 3)
            self.metrics["kstat-4"].value = scipy.stats.kstat(scalar_data, 4)
            self.metrics["kstatvar"].value = scipy.stats.kstatvar(scalar_data)
            self.metrics["iqr"].value = scipy.stats.iqr(scalar_data)
            bMean, bVar, bStd = scipy.stats.bayes_mvs(scalar_data)

            # Singular Valued Based
            self.metrics["bayes-mean"].value = bMean[1]
            self.metrics["bayes-mean-avg"].value = bMean[0]
            self.metrics["bayes-var"].value = bVar[1]
            self.metrics["bayes-var-avg"].value = bVar[0]
            self.metrics["bayes-std"].value = bStd[1]
            self.metrics["bayes-std-avg"].value = bStd[0]

def _get_scalar_data(X, mask):
    result = np.copy(X)
    i = len(mask)-1
    while i >= 0:
        if mask[i] == 1:
            result = np.delete(result, i, axis=1)
        i = i-1
    return result