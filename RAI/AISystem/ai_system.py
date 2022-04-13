from math import exp
import pandas as pd
from typing import Any
import numpy as np
import RAI
from RAI.AISystem.task import Task
from RAI.dataset.dataset import Data, Dataset, MetaDatabase
from RAI.certificates import CertificateManager
from RAI.metrics import MetricManager

 

class AISystem:
    def __init__(self,
                meta_database:MetaDatabase, 
                dataset:Dataset, 
                task:Task,
                enable_certificates:bool = True,
                custom_certificate_location:str = None ) -> None:
        
        # if type(user_config) is not dict:
        #     raise TypeError("User config must be of type Dictionary")
        
        self.meta_database = meta_database
        self.task = task
        self.dataset = dataset
        # self.user_config = user_config
        self.enable_certificates = enable_certificates
        
        
        
        self._timestamp = ""
        self._sample_count = 0
        
        self._last_metric_values = None
        self._last_certificate_values=None
    
        self.metric_manager = MetricManager(self)
        self.certificate_manager = CertificateManager()
        if custom_certificate_location is None:
            self.certificate_manager.load_stock_certificates()
        else:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)


    def initialize(self, user_config:dict, **kw_args):
        self.metric_manager.initialize( user_config, *kw_args)    
     
    
    def get_metric_values(self) -> dict:
        return self._last_metric_values

    def get_certificate_values(self) -> dict:
        return self._last_certificate_values

 

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
    
    def compute(self, predictions: np.ndarray, data_type:str = "train") -> None:
        
        
        data_dict = {"data": self.get_data(data_type)}
        if predictions is not None:
            data_dict["predictions"] = predictions
       
       
        self._last_metric_values = self.metric_manager.compute( data_dict )
        
        if self.enable_certificates:
            self._last_certificate_values = self.certificate_manager.compute(self._last_metric_values)

    def get_metric_info(self):
        return self.metric_manager.get_metadata()

    def get_certificate_info(self):
        return self.certificate_manager.get_metadata()
        
 
    # we have not implemented the incremental update as of now and each call to compute process all the data
    def update(self, data):
        raise NotImplemented()
     
 
    def set_agent(self, agent):
        self.task.model.agent = agent