from abc import ABC, abstractmethod


class DisplayElement(ABC):
    def __init__(self, name):
        self.requires_tag_chooser = False
        self._name = name
        self._data = {}
        self._metric_object = {}

    @abstractmethod
    def append(self, data, tag):
        pass

    @abstractmethod
    def to_string(self):
        pass

    def to_display(self):
        pass
