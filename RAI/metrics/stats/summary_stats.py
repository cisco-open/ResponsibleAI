from RAI.metrics.metric_group import MetricGroup
import numpy as np
import scipy.stats
import warnings
import os


class StatMetricGroup(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        
    def update(self, data):
        pass

    def compute(self, data_dict):
        if "data" in data_dict:
            args = {}
            if self.ai_system.metric_manager.user_config is not None and "stats" in self.ai_system.metric_manager.user_config and "args" in self.ai_system.metric_manager.user_config["stats"]:
                args = self.ai_system.metric_manager.user_config["stats"]["args"]
            data = data_dict["data"]

            scalar_data = data.scalar
            self.metrics["mean"].value = np.mean(scalar_data, **args.get("mean", {}), axis=0)
            self.metrics["covariance"].value = np.cov(scalar_data.T, **args.get("covariance", {}))
            self.metrics["num-Nan-rows"].value = np.count_nonzero(np.isnan(data.X).any(axis=1))
            self.metrics["percent-Nan-rows"].value = self.metrics["num-Nan-rows"].value/np.shape(np.asarray(data.X))[0]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.metrics["geometric-mean"].value = scipy.stats.mstats.gmean(scalar_data)
            
            self.metrics["mode"].value = scipy.stats.mstats.mode(scalar_data)[0][0]
            self.metrics["skew"].value = scipy.stats.mstats.skew(scalar_data)
            self.metrics["variation"].value = scipy.stats.mstats.variation(scalar_data)
            self.metrics["sem"].value = scipy.stats.mstats.sem(scalar_data)
            self.metrics['kurtosis'].value = scipy.stats.mstats.kurtosis(scalar_data)

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
