from math import exp
import pandas as pd
import datetime
import json
from typing import Any
from RAI import utils
import RAI
import numpy as np
from RAI.AISystem.task import Task
from RAI.dataset.dataset import Data, Dataset, MetaDatabase
from RAI.metrics.registry import registry
from RAI.certificates import CertificateManager

 

class AISystem:
    def __init__(self,
                meta_database:MetaDatabase, 
                dataset:Dataset, 
                task:Task,
                user_config:dict, 
                enable_certificates:bool = True,
                custom_certificate_location:str = None ) -> None:
        
        if type(user_config) is not dict:
            raise TypeError("User config must be of type Dictionary")
        
        self.meta_database = meta_database
        self.task = task
        self.dataset = dataset
        self.user_config = user_config
        self.enable_certificates = enable_certificates
        
        
        self.metric_groups = {}
        self._timestamp = ""
        self._sample_count = 0
        self._last_metric_values = None
        self._last_certificate_values=None

        self.certificate_manager = CertificateManager()
        if custom_certificate_location is None:
            self.certificate_manager.load_stock_certificates()
        else:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)


        
    
    def initialize(self, metric_groups:list[str] = None, max_complexity:str = "linear"):
        for metric_group_name in registry:
            if metric_groups is not None and metric_group_name not in metric_groups:
                continue

            metric_class = registry[metric_group_name]
            if metric_class.is_compatible(self):
                # if self._is_compatible(temp.compatibility):
                self.metric_groups[metric_group_name] = metric_class(self)
                print( f"metric group : {metric_group_name} was loaded" )


    
    def get_metric_values(self) -> dict:
        return self._last_metric_values

    def get_certificate_values(self) -> dict:
        return self._last_certificate_values


    def get_metric(self, metric_group_name:str, metric_name:str) -> Any :
        print(f"request for metric group : {metric_group_name}, metric_name : {metric_name}")
        return self.metric_groups[metric_group_name].metrics[metric_name].value

    def reset_measurements(self) -> None:
        for metric_group_name in self.metric_groups:
           self.metric_groups[metric_group_name].reset()
        

        self._last_certificate_values = None
        self._last_metric_values = None
        self._sample_count = 0
        self._time_stamp = None  # Replace by registering a time metric in metric_groups? 

    def get_data(self, data_type:str) ->  Data :
        if data_type == "train":
            return self.dataset.train_data
        if data_type == "val":
            return self.dataset.val_data
        if data_type == "test":
            return self.dataset.test_data
        raise Exception(f"unknown data type : {data_type}" )

    def get_model_info(self) -> dict :
        result = {"id": self.task.model.name, "model": self.task.model.model_class, "adaptive": self.task.model.adaptive,
                  "task_type": self.task.type, "configuration": self.user_config, "features": [], "description": self.task.description,
                  "display_name": self.task.model.display_name}
        for i in range(len(self.meta_database.features)):
            result['features'].append(self.meta_database.features[i].name)
        return result
 
    def get_metric_info_flat(self) -> dict :
        result = {}
        for group in self.metric_groups:
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[metric_obj.unique_name] = metric_obj.config
                metric_obj.config["tags"] = self.metric_groups[group].tags  # Change this up after
        return result
    

    
    def compute(self, predictions: np.ndarray, data_type:str = "train") -> None:
        
        data_dict = {"data": self.get_data(data_type)}
        if predictions is not None:
            data_dict["predictions"] = predictions
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].compute(data_dict)
        
        self._timestamp = self._get_time()
        
        result = {}
        for group in self.metric_groups:
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[metric_obj.unique_name] = utils.jsonify(metric_obj.value)
               
        self._last_metric_values =  result
        
        if self.enable_certificates:
            self._last_certificate_values = self.certificate_manager.compute(self._last_metric_values)

        self._sample_count += len(data_dict)
        
 

    def update_metrics(self, data):
        raise NotImplemented()
        # for i in range(len(data)):
        #     for metric_group_name in self.metric_groups:
        #         self.metric_groups[metric_group_name].update(data[i])
        # self.timestamp = self._get_time()
        # self.sample_count += 1
 

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)
    

    def _dict_to_csv(self, file:str, dictionary:dict, write_headers:bool = True) -> None:
        newDict = {}
        newDict['date'] = self.timestamp
        for category in dictionary:
            for metric in dictionary[category]:
                newDict[metric] = dictionary[category][metric]
        df = pd.DataFrame([newDict])
        df.to_csv(file, header=write_headers, mode='a', index=False)
 

    # Searches all metrics. Queries based on Metric Name, Metric Group Name, Category, and Tags.
    def search(self, query:str) -> dict :
        query = query.lower()
        results = {}
        for group in self.metric_groups:
            add_group = group.lower() == query 
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                if add_group or metric.lower().find(query) > -1 or metric_obj.display_name.lower().find(query) > -1:
                    results[metric] = metric_obj.value
                elif metric_obj.tags is not None:
                    for tag in metric_obj.tags:
                        if tag.lower().find(query) > -1:
                            results[metric] = metric_obj.value
                            break
        return results


    
 
    def set_agent(self, agent):
        self.task.model.agent = agent