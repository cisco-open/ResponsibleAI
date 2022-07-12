from abc import ABC, abstractmethod
from .display_registry import register_class


class DisplayElement(ABC):
    def __init__(self, name):
        self.requires_tag_chooser = False
        self.requirements = []
        self._name = name
        self._data = {}
        self._metric_object = {}

    def __init_subclass__(cls, class_location=None, **kwargs):
        super().__init_subclass__(**kwargs)
        register_class(cls.__name__, cls)

    @abstractmethod
    def append(self, data, tag):
        pass

    @abstractmethod
    def to_string(self):
        pass

    def to_display(self):
        pass
