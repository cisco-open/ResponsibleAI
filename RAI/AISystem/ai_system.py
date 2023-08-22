# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy


from RAI.AISystem.model import Model
from RAI.certificates import CertificateManager
from RAI.dataset.dataset import Data, NumpyData, IteratorData, Dataset, MetaDatabase
from RAI.metrics import MetricManager
from RAI.all_types import all_output_requirements, all_task_types, all_metric_types


class AISystem:
    """
    AI Systems are the main class users interact with in RAI.
    When constructed, AISystems are passed a name, a task type, a MetaDatabase, a Dataset and a Model.

    :param name: Create a new string object from the given object.
     If encoding or errors is specified, then the object must expose a data buffer that will be decoded using the given encoding and error handler. Otherwise, returns the result of object.__str__() (if defined) or repr(object). encoding defaults to sys.getdefaultencoding(). errors defaults to 'strict'.
    :param task: Create a new string object from the given object. If encoding or errors is specified, then the object must expose a data buffer that will be decoded using the given encoding and error handler. Otherwise, returns the result of object.__str__() (if defined) or repr(object). encoding defaults to sys.getdefaultencoding(). errors defaults to 'strict'.
    :param meta_database: The RAI MetaDatabase class holds Meta information about the Dataset. It includes information about the features, and contains maps and masks to quick get access to the different feature data of different information.
    :param dataset: The RAI Dataset class holds a dictionary of RAI Data classes, for example {'train': trainData, 'test': testData}, where trainData and testData are RAI Data objects.
    :param model: Model is RAIs abstraction for the ML Model performing inferences. When constructed, models are optionally passed the name, the models functions for inferences, its name, the model, its optimizer, its loss function, its class and a description. Attributes of the model are used to determine which metrics are relevant.
    :param enable_certificates: Returns True when the argument x is true, False otherwise. The builtins True and False are the only two instances of the class bool. The class bool is a subclass of the class int, and cannot be subclassed.

    """  # noqa: E501

    def __init__(self,
                 name: str,
                 task: str,
                 meta_database: MetaDatabase,
                 dataset: Dataset,
                 model: Model,
                 enable_certificates: bool = True) -> None:
        assert task in all_task_types, "Task must be in " + str(all_task_types)
        self.name = name
        self.task = task
        self.meta_database = meta_database
        self.model = model
        self.dataset = dataset
        self.enable_certificates = enable_certificates
        self.auto_id = 0
        self._last_metric_values = {}
        self._last_certificate_values = None
        self.metric_manager = None
        self.certificate_manager = None
        self.data_summarizer = None
        self.user_config = None
        self.data_dict = {}
        self.custom_metrics = {}
        self.custom_functions = []

    def initialize(self, user_config: dict = {},
                   custom_certificate_location: str = None,
                   custom_metrics: dict = {},
                   custom_functions: list = None
                   ):
        """
        :param user_config: Takes user config as a dict
        :param custom_certificate_location: certificate path by default it is None
        :param custom_metrics: dict of custom metrics you want to display on the dashboard
        :param custom_functions: list of custom functions that take the existing metrics as input and return a value

        :return: None
        """
        self.user_config = user_config
        self.custom_metrics = custom_metrics
        self.custom_functions = custom_functions
        masks = {"scalar": self.meta_database.scalar_mask, "categorical": self.meta_database.categorical_mask,
                 "image": self.meta_database.image_mask, "text": self.meta_database.text_mask}
        self.dataset.separate_data(masks)
        self.meta_database.initialize_requirements(self.dataset, "fairness" in user_config)
        self.metric_manager = MetricManager(self)
        self.certificate_manager = CertificateManager()
        self.certificate_manager.load_stock_certificates()
        if custom_certificate_location is not None:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)

    def get_metric_values(self) -> dict:
        """
        Returns the last metric values in the form of key value pair

        :param self: None
        :return: last metric values(dict)

        """
        return self._last_metric_values

    def add_certificates(self):
        """
        Add certificates values to the existing metrics
        :return: None
        """
        certificates = self.get_certificate_values()
        if not certificates:
            return
        certificate_info = self.get_certificate_info()
        for metric_group in self._last_metric_values:
            self._last_metric_values[metric_group]['Certificates'] = {}
            for certificate in certificates:
                name = certificate_info.get(certificate, {}).get('display_name', '-').title()
                self._last_metric_values[metric_group]['Certificates'][name] = certificates[certificate]['value']

    def add_custom_metrics(self):
        """
        Add custom metrics to existing metrics

        :return: None
        """
        class ActiveMetrics(object):
            def __init__(self, group):
                self.group = group

            def __getattr__(self, item):
                raise AttributeError(f'{self.group} group has no attribute {item}')

        if not any([self.custom_metrics, self.custom_functions]):
            return

        for metric_group in self._last_metric_values:
            self._last_metric_values[metric_group]['Custom'] = {}
            for metric, metrics_values in self._last_metric_values[metric_group].items():
                exec(f'{metric} = ActiveMetrics("{metric}")')
                for individual_metric, individual_metric_value in metrics_values.items():
                    try:
                        exec(f'setattr({metric}, "{individual_metric}", {individual_metric_value})')
                    except Exception:
                        pass
            for custom_metric, custom_metric_expression in self.custom_metrics.items():
                try:
                    self._last_metric_values[metric_group]['Custom'][custom_metric] = eval(custom_metric_expression)
                except Exception as e:
                    print(f'\nUnable to calculate custom metric {custom_metric}.\n'
                          f'Make sure to use the correct metric with the correct group association.\n')
                    raise e
            if self.custom_functions:
                data = deepcopy(self._last_metric_values[metric_group])
                for func in self.custom_functions:
                    try:
                        self._last_metric_values[metric_group]['Custom'][func.__name__] = func(data)
                    except Exception as e:
                        print(f'\nUnable to calculate custom metric from {func.__name__}.\n'
                              f'Make sure to implement a function that takes a dict parameter as input.\n')
                        raise e

    def display_metric_values(self, display_detailed: bool = False):
        """
        :param display_detailed: if True we need to display metric explanation if False we don't have to display

        :return: None

        Displays the metric values
        """  # noqa: E501
        vals = self._last_metric_values
        info = self.get_metric_info()
        for dataset in vals:
            print("\n\n===== " + dataset + " Dataset =====")
            for group in vals[dataset]:
                print("\n----- " + info[group]['meta']["display_name"] + " Metrics -----")
                for metric in vals[dataset][group]:
                    if info[group][metric]["type"] in all_metric_types:
                        print(info[group][metric]["display_name"] + ": ", vals[dataset][group][metric])
                        if display_detailed:
                            print(f'{info[group][metric]["display_name"]} is {info[group][metric]["explanation"]}\n')

    def get_certificate_values(self) -> dict:
        """
        Returns the last used certificate information

        :param self:  None

        :return: Certificate infomation(dict)
         Returns the last used certificate information
        """
        return self._last_certificate_values

    def get_data(self, data_type: str) -> Data:
        """
        get_data accepts data_type and returns the data object information

        :param data_type(str):  Get the data type information

        :return: Dataset datatype information(str)

        """
        return self.dataset.data_dict.get(data_type, None)

    def get_project_info(self) -> dict:
        """
        Fetch the project information like name, configuration, metric user config and Returns the project details

        :param self:  None

        :return: Project details(dict)

        """
        result = {"id": self.name,
                  "task_type": self.task, "configuration": self.metric_manager.user_config, "features": [],
                  "description": self.model.description, "output_features": []}
        for i in range(len(self.meta_database.features)):
            result['features'].append(self.meta_database.features[i].name)
        for i in range(len(self.model.output_features)):
            result['output_features'].append(self.model.output_features[i].name)
        return result

    def get_data_summary(self) -> dict:

        """
        process the data and returns the summary consisting of prediction, label details

        :param self:  None

        :return: Data Summary(Dict)

        """

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
        key = data_type if data_type is not None else "No Dataset"
        if isinstance(data_dict["data"], NumpyData):
            self._last_metric_values[key] = \
                self.metric_manager.compute(data_dict)
        elif isinstance(data_dict["data"], IteratorData):
            self._last_metric_values[key] = \
                self.metric_manager.iterator_compute(data_dict, predictions)
        if self.enable_certificates:
            self._last_certificate_values = self.certificate_manager.compute(self._last_metric_values.get(key))

    # Compute will tell RAI to compute metric values across each dataset which predictions were made on.
    def compute(self, predictions: dict, tag=None) -> None:

        """
        Compute will tell RAI to compute metric values across each dataset which predictions were made on

        :param predictions(dict): Prediction value from the classifier
        :param tag: by default None
        :return: None
        """
        self._last_metric_values = {}
        if len(self.dataset.data_dict) == 0:  # Model with no X, y data.
            for key in predictions.keys():
                self._single_compute(predictions, None, tag=tag)
                return
        elif not (isinstance(predictions, dict) and all(isinstance(v, dict) for v in predictions.values())
                  and all(isinstance(k, str) for k in predictions.keys())):  # noqa: W503
            raise Exception("Prediction dictionary should be in the form [dataset][output_type] -> nd.array")
        for key in predictions.keys():
            if key in self.dataset.data_dict.keys():
                self._single_compute(predictions[key], key, tag=tag)

        self.add_certificates()
        self.add_custom_metrics()

    # Run Compute automatically generates outputs from the model, and compute metrics based on those outputs
    def run_compute(self, tag=None) -> None:
        """
        Run Compute automatically generates outputs from the model, and compute metrics based on those outputs

        :param tag: tag by default None or we can pass model as a string

        :return: Data Summary(Dict)

        """
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
        """
        Returns the metadata of the metric_manager class

        :param self: None
        :return: metric Manager metadata

        """
        return self.metric_manager.get_metadata()

    def get_certificate_info(self):
        """
        Returns the metadata of the certificate_manager class

        :param self: None

        :return: Certificate info from certificate_manager

        """
        return self.certificate_manager.get_metadata()

    # we have not implemented the incremental update as of now and each call to compute process all the data
    def update(self, data):
        raise NotImplementedError
