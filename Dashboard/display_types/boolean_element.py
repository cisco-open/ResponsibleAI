from display_object import DisplayElement


# For single valued boolean metrics
class BooleanElement(DisplayElement):
    def __init__(self, data):
        assert isinstance(data, bool), "Data must be of type bool"
        super().__init__(data)

    def to_string(self):
        print(self._data)

    def to_display(self):
        pass



