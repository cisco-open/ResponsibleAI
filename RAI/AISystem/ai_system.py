import numpy as np
from RAI.AISystem.model import Model
from RAI.dataset.dataset import Data, Dataset, MetaDatabase
from RAI.certificates import CertificateManager
from RAI.metrics import MetricManager


class AISystem:
    def __init__(self,
                name:str,
                meta_database:MetaDatabase, 
                dataset:Dataset, 
                model:Model,
                enable_certificates:bool = True,
                ) -> None:
        
        # if type(user_config) is not dict:
        #     raise TypeError("User config must be of type Dictionary")
        
        self.name = name
        self.meta_database = meta_database
        self.model = model
        self.dataset = dataset
        # self.user_config = user_config
        self.enable_certificates = enable_certificates
        self.auto_id = 0
        self._last_metric_values = None
        self._last_certificate_values=None

    def initialize(self, user_config:dict, custom_certificate_location:str = None , **kw_args):
        self.metric_manager = MetricManager(self)
        self.certificate_manager = CertificateManager()
        
        self.certificate_manager.load_stock_certificates()
        if custom_certificate_location is not None:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)
        self.metric_manager.initialize(user_config, *kw_args)
        self.dataset.separate_data(self.meta_database.scalar_mask)

    def get_metric_values(self) -> dict:
        return self._last_metric_values

    def get_certificate_values(self) -> dict:
        return self._last_certificate_values

    def get_data(self, data_type:str) -> Data:
        if data_type == "train":
            return self.dataset.train_data
        if data_type == "val":
            return self.dataset.val_data
        if data_type == "test":
            return self.dataset.test_data
        raise Exception(f"unknown data type : {data_type}" )

    def get_project_info(self) -> dict :
        result = {"id": self.name,  
                  "task_type": self.task.type, "configuration": self.metric_manager.user_config, "features": [], "description": self.task.description,
                  }
        for i in range(len(self.meta_database.features)):
            result['features'].append(self.meta_database.features[i].name)
        return result
    
    def compute(self, predictions: np.ndarray, data_type:str = "test", tag=None) -> None:
        self.auto_id += 1
        if tag is None:
            tag = f"{self.auto_id}"
        data_dict = {"data": self.get_data(data_type)}
        if predictions is not None:
            data_dict["predictions"] = predictions
        data_dict["tag"] = tag
       
        self._last_metric_values = self.metric_manager.compute(data_dict)
        
        if self.enable_certificates:
            self._last_certificate_values = self.certificate_manager.compute(self._last_metric_values)

    def get_metric_info(self):
        return self.metric_manager.get_metadata()

    def get_certificate_info(self):
        return self.certificate_manager.get_metadata()

    # we have not implemented the incremental update as of now and each call to compute process all the data
    def update(self, data):
        raise NotImplemented()
