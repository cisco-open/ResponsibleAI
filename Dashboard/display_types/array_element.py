from display_object import DisplayElement
import numpy as np

# For metrics which return arrays unrelated to features
class ArrayElement(DisplayElement):
    def __init__(self, data):
        assert type(data) in [np.ndarray, list], "data must be of type np.ndarray or list"
        super().__init__(data)

    def to_string(self):
        print(self._data)

    def to_display(self):
        pass
