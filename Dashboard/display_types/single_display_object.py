from abc import ABCMeta, abstractmethod
from .display_object import DisplayElement


# Single display elements are display elements that can only be shown one at a time
# For example, matrices and images.
class SingleDisplayElement(DisplayElement, metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__(name)
        self.requires_tag_chooser = True

    def get_tags(self):
        return self._data["tag"]

    @abstractmethod
    def display_tag_num(self, tag_num):
        pass
