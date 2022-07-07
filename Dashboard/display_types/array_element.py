from display_object import DisplayElement
import numpy as np


# TODO: Implement once we have an array element
# For metrics which return arrays unrelated to features
class ArrayElement(DisplayElement):
    def __init__(self, name):
        super().__init__(name)

    def to_string(self):
        print(self._data)

    def to_display(self):
        pass
