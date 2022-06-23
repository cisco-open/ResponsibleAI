import numpy as np
import time
from RAI.AISystem.model import Model
from RAI.dataset.dataset import Data, Dataset, MetaDatabase
from RAI.certificates import CertificateManager
from RAI.metrics import MetricManager


class AISystem:
    def __init__(self,
                 name: str,
                 meta_database: MetaDatabase,
                 dataset: Dataset,
                 model: Model,
                 enable_certificates: bool = True) -> None:

        self.name = name
        self.meta_database = meta_database
        self.model = model
        self.dataset = dataset
        self.enable_certificates = enable_certificates
        self.auto_id = 0
        self._last_metric_values = {}
        self._last_certificate_values = None
        self.metric_manager = None
        self.certificate_manager = None

    def initialize(self, user_config: dict, custom_certificate_location: str = None, **kw_args):
        self.dataset.separate_data(self.meta_database.scalar_mask)
        self.meta_database.initialize_requirements(list(self.dataset.data_dict.values())[0], "fairness" in user_config)
        self.metric_manager = MetricManager(self)
        self.certificate_manager = CertificateManager()
        self.certificate_manager.load_stock_certificates()
        if custom_certificate_location is not None:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)
        self.metric_manager.initialize(user_config, *kw_args)

    def get_metric_values(self) -> dict:
        return self._last_metric_values

    def get_certificate_values(self) -> dict:
        return self._last_certificate_values

    def get_data(self, data_type:str) -> Data:
        if data_type not in self.dataset.data_dict:
            raise Exception(f"data_type must be found in Dataset. Got : {data_type}")
        return self.dataset.data_dict[data_type]

    def get_project_info(self) -> dict :
        result = {"id": self.name,  
                  "task_type": self.model.task, "configuration": self.metric_manager.user_config, "features": [], "description": self.model.description,
                  }
        for i in range(len(self.meta_database.features)):
            result['features'].append(self.meta_database.features[i].name)
        return result
    
    def single_compute(self, predictions: np.ndarray, data_type: str = "test", tag=None) -> None:
        self.auto_id += 1
        if tag is None:
            tag = f"{self.auto_id}"
        data_dict = {"data": self.get_data(data_type)}
        if predictions is not None:
            data_dict["predictions"] = predictions
        data_dict["tag"] = tag
        self._last_metric_values[data_type] = self.metric_manager.compute(data_dict)
        if self.enable_certificates:
            self._last_certificate_values = self.certificate_manager.compute(self._last_metric_values)

    def compute(self, predictions: dict, tag=None) -> None:
        if not (isinstance(predictions, dict) and all(isinstance(v, np.ndarray) for v in predictions.values()) \
                and all(isinstance(k, str) for k in predictions.keys())):
            raise Exception("Predictions should be a dictionary of strings mapping to np.ndarrays")
        for key in predictions.keys():
            if key in self.dataset.data_dict.keys():
                self.single_compute(predictions[key], key, tag=tag)

    def run_compute(self, tag=None) -> None:
        # TODO: Generalize across all model functions, different model types
        # Prediction generation and computation must be separated due to some weird sklearn bug
        preds = {}
        for category in self.dataset.data_dict:
            data = self.dataset.data_dict[category].X
            preds[category] = self.model.predict_fun(data)
        for key in preds:
            self.single_compute(preds[key], key)

    def get_metric_info(self):
        return self.metric_manager.get_metadata()

    def get_certificate_info(self):
        return self.certificate_manager.get_metadata()

    # we have not implemented the incremental update as of now and each call to compute process all the data
    def update(self, data):
        raise NotImplemented()
