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
        self.max_progress_tick = 5
        self.current_tick = 0
        self.connection = None # connection is an optional function passed to share progress with dashboard
        print("Analysis created")

    def __init_subclass__(cls, class_location=None, **kwargs):
        super().__init_subclass__(**kwargs)
        config_file = class_location[:-2] + "json"
        cls.config = json.load(open(config_file))
        register_class(cls.__name__, cls)

    @classmethod
    def is_compatible(cls, ai_system, dataset: str):
        compatible = cls.config["compatibility"]["task_type"] is None or cls.config["compatibility"]["task_type"] == [] \
                     or ai_system.task in cls.config["compatibility"]["task_type"] \
                     or ("classification" in cls.config["compatibility"]["task_type"]
                         and ai_system.task == "binary_classification")
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

    def progress_percent(self, percentage_complete):
        percentage_complete = int(percentage_complete)
        if self.conncetion is not None:
            self.connection(str(percentage_complete))

    def progress_tick(self):
        self.current_tick += 1
        percentage_complete = min(100, int(100*self.current_tick/self.max_progress_tick))
        if self.connection is not None:
            self.connection(str(percentage_complete))

    # connection is a function that accepts progress, and pings the dashboard
    def set_connection(self, connection):
        self.connection = connection

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def to_html(self):
        pass
