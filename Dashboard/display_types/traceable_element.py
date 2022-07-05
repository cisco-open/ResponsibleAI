from abc import ABC, abstractmethod
from .display_object import DisplayElement


class TraceableElement(DisplayElement):
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def add_trace_to(self, fig):
        pass
