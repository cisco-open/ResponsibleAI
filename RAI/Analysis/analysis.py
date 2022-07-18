from RAI.AISystem import AISystem
from abc import ABC, abstractmethod
from .analysis_registry import register_class
from RAI.utils import compare_runtimes
import json


class Analysis(ABC):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str=None):
        self.ai_system = ai_system
        self.analysis = {}
        self.dataset = dataset
        self.tag = None
        print("Analysis created")

    def __init_subclass__(cls, class_location=None, **kwargs):
        super().__init_subclass__(**kwargs)
        config_file = class_location[:-2] + "json"
        cls.config = json.load(open(config_file))
        register_class(cls.__name__, cls)

    @classmethod
    def is_compatible(cls, ai_system, dataset: str):
        compatible = cls.config["compatibility"]["task_type"] is None or cls.config["compatibility"]["task_type"] == "" \
                     or cls.config["compatibility"]["task_type"] == ai_system.task \
                     or (cls.config["compatibility"][
                             "task_type"] == "classification" and ai_system.task == "binary_classification")
        compatible = compatible and (cls.config["compatibility"]["data_type"] is None or cls.config["compatibility"][
            "data_type"] == [] or all(item in ai_system.meta_database.data_format
                                      for item in cls.config["compatibility"]["data_type"]))
        compatible = compatible and (cls.config["compatibility"]["output_requirements"] is None or
                                     all(item in ai_system.data_dict for item in
                                         cls.config["compatibility"]["output_requirements"]))
        compatible = compatible and (cls.config["compatibility"]["dataset_requirements"] is None or
                                     all(item in ai_system.meta_database.stored_data for item in
                                         cls.config["compatibility"]["dataset_requirements"]))
        compatible = compatible and compare_runtimes(ai_system.metric_manager.user_config.get("time_complexity"),
                                                     cls.config["complexity_class"])
        compatible = compatible and all(group in ai_system.get_metric_values()[dataset]
                                        for group in cls.config["compatibility"]["required_groups"])
        return compatible

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def to_html(self):
        pass
