from RAI.AISystem.model import Model
from RAI.certificates import CertificateManager
from RAI.dataset.dataset import Data, Dataset, MetaDatabase
from RAI.metrics import MetricManager
from RAI.all_types import all_output_requirements, all_task_types, all_metric_types
from RAI.dataset.vis import DataSummarizer
from RAI.interpretation.interpreter import Interpreter
import numpy as np

class AISystem:
    """
    AI Systems are the main class users interact with in RAI.
    When constructed, AISystems are passed a name, a task type, a MetaDatabase, a Dataset and a Model.
    """

    def __init__(self,
                 name: str,
                 task: str,
                 meta_database: MetaDatabase,
                 dataset: Dataset,
                 model: Model,
                 interpret_methods: list[str] = [],
                 enable_certificates: bool = True) -> None:
        assert task in all_task_types, "Task must be in " + str(all_task_types)
        self.name = name
        self.task = task
        self.meta_database = meta_database
        self.model = model
        self.dataset = dataset
        self.interpret_methods = interpret_methods
        self.enable_certificates = enable_certificates
        self.auto_id = 0
        self._last_metric_values = {}
        self._last_certificate_values = None
        self.metric_manager = None
        self.certificate_manager = None
        self.data_summarizer = None
        self.user_config = None
        self.data_dict = {}

    def initialize(self, user_config: dict, custom_certificate_location: str = None, **kw_args):
        self.user_config = user_config
        self.dataset.separate_data(self.meta_database.scalar_mask, self.meta_database.categorical_mask,
                                   self.meta_database.image_mask, self.meta_database.text_mask)
        self.meta_database.initialize_requirements(self.dataset, "fairness" in user_config)
        self.metric_manager = MetricManager(self)
        self.certificate_manager = CertificateManager()
        self.certificate_manager.load_stock_certificates()
        if custom_certificate_location is not None:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)
        # self.data_summarizer = DataSummarizer(self.dataset, self.model.output_features[0].possibleValues, self.task)
        self.interpreter = Interpreter(self.interpret_methods, self.model, self.dataset)
        print("output features: ", self.model.output_features)
        self.data_summarizer = DataSummarizer(self.dataset, self.task, self.model.output_features)

    def get_metric_values(self) -> dict:
        return self._last_metric_values

    def display_metric_values(self, display_detailed=False):
        vals = self._last_metric_values
        info = self.get_metric_info()
        for dataset in vals:
            print("\n\n===== " + dataset + " Dataset =====")
            for group in vals[dataset]:
                print("\n----- " + info[group]['meta']["display_name"] + " Metrics -----")
                for metric in vals[dataset][group]:
                    if(info[group][metric]["type"] in all_metric_types):
                        print(info[group][metric]["display_name"] + ": ", vals[dataset][group][metric])
                        if display_detailed:
                            print(info[group][metric]["display_name"] + " is " + info[group][metric]["explanation"], "\n")

    def get_certificate_values(self) -> dict:
        return self._last_certificate_values

    def get_data(self, data_type: str) -> Data:
        return self.dataset.data_dict.get(data_type, None)

    def get_project_info(self) -> dict:
        result = {"id": self.name,
                  "task_type": self.task, "configuration": self.metric_manager.user_config, "features": [],
                  "description": self.model.description, "output_features": []}
        for i in range(len(self.meta_database.features)):
            result['features'].append(self.meta_database.features[i].name)
        for i in range(len(self.model.output_features)):
            result['output_features'].append(self.model.output_features[i].name)
        return result

    def get_data_summary(self) -> dict:
        pred_target = self.data_summarizer.target 
        label_name = self.data_summarizer.label_name_dict
        labels = self.data_summarizer.labels 
        label_dist_dict = self.data_summarizer.getLabelDistribution()
        if label_name is None:
            label_name = ""
        # TODO: Add data sampler
        summary = {
            "pred_target": pred_target,
            "label_name": label_name,
            "labels": labels,
            "label_dist": label_dist_dict,
        }
        return summary 

    def get_interpretation(self) -> dict:
        interpretation = self.interpreter.getModelInterpretation()
        return interpretation 
    
    def _single_compute(self, predictions: dict, data_type: str = "test", tag=None) -> None:
    # Single compute accepts predictions and the name of a dataset, and then calculates metrics for that dataset.
        self.auto_id += 1
        if tag is None:
            tag = f"{self.auto_id}"
        data_dict = {"data": self.get_data(data_type)}
        for output_type in all_output_requirements:
            if output_type in predictions:
                data_dict[output_type] = predictions[output_type]
            elif output_type in data_dict:
                data_dict.pop(output_type)
        data_dict["tag"] = tag
        self.data_dict = data_dict
        self.metric_manager.initialize(self.user_config)
        self._last_metric_values[data_type if data_type is not None else "No Dataset"] = self.metric_manager.compute(data_dict)
        if self.enable_certificates:
            self._last_certificate_values = self.certificate_manager.compute(self._last_metric_values)

    # Compute will tell RAI to compute metric values across each dataset which predictions were made on.
    def compute(self, predictions: dict, tag=None) -> None:
        self._last_metric_values = {}
        if len(self.dataset.data_dict) == 0:  # Model with no X, y data.
            for key in predictions.keys():
                self._single_compute(predictions, None, tag=tag)
                return
        elif not (isinstance(predictions, dict) and all(isinstance(v, dict) for v in predictions.values()) \
                and all(isinstance(k, str) for k in predictions.keys())):
            raise Exception("Prediction dictionary should be in the form [dataset][output_type] -> nd.array")
        for key in predictions.keys():
            if key in self.dataset.data_dict.keys():
                self._single_compute(predictions[key], key, tag=tag)

    # Run Compute automatically generates outputs from the model, and compute metrics based on those outputs
    def run_compute(self, tag=None) -> None:
        self._last_metric_values = {}
        preds = {}
        for category in self.dataset.data_dict:
            preds[category] = {}
            data = self.dataset.data_dict[category].X
            for function_type in self.model.output_types:
                preds[category][function_type] = self.model.output_types[function_type](data)
        for key in preds:
            self._single_compute(preds[key], key)

    def get_metric_info(self):
        return self.metric_manager.get_metadata()

    def get_certificate_info(self):
        return self.certificate_manager.get_metadata()

    # we have not implemented the incremental update as of now and each call to compute process all the data
    def update(self, data):
        raise NotImplemented()
