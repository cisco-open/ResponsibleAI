import numpy as np
import pandas as pd
import sklearn as sk
import time
from RAI.metrics.registry import registry
from .task import *
from RAI.dataset import *


class AISystem:
    def __init__(self, meta_database, dataset, task) -> None:
        self.meta_database = meta_database
        self.task = task
        self.dataset = dataset
        self.metric_groups = {}
        self.timestamp = None
        self.sample_count = 0


    def initialize(self,
                   metric_groups = None,
                   metric_group_re = None,
                   max_complexity = "linear" 
                   ):
        
        for metric_group_name in registry:
            self.metric_groups[metric_group_name] = registry[metric_group_name](self)
            print("metric group : {} was created".format(metric_group_name))


# May be more convinient to just accept metric name (or add funcitonality to detect group names and return a dictionary)
    def get_metric(self, metric_group_name, metric_name): 
        print("request for metric group : {}, metric_name : {}".format(metric_group_name,metric_name))
        return self.metric_groups[metric_group_name].metrics[metric_name].value


    def reset_metrics(self):
        for metric_group_name in self.metric_groups:
           self.metric_groups[metric_group_name].reset()
        self.sample_count = 0
        self.time_stamp = None  # Replace by registering a time metric in metric_groups? 


    def get_data(self, data_type) :
        if data_type == "train":
            return self.dataset.train_data
        if data_type == "val":
            return self.dataset.val_data
        if data_type == "test":
            return self.dataset.test_data
        raise Exception("unknown data type : {}".format(data_type))


    def compute_metrics(self, reset_metrics=False, data_type = "train"):
        if reset_metrics:
            self.reset_metrics()
        
        data = self.get_data(data_type)
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].compute(data)
        self.time_stamp = time.time()
        self.sample_count += len(data)


    def update_metrics(self, data):
        for i in range(len(data)):
            for metric_group_name in self.metric_groups:
                self.metric_groups[metric_group_name].update(data[i])
        self.time_stamp = time.time()
        self.sample_count += 1


    def export_metrics_values(self):
        result = {}
        for metric_group_name in self.metric_groups:
            result[metric_group_name] = self.metric_groups[metric_group_name].export_metrics_values()
        return result





