from display_object import DisplayElement


# For metrics which return arrays unrelated to features
class ImageElement(DisplayElement):
    def __init__(self, data):
        assert isinstance(data, dict), "data must be of type dict"
        super().__init__(data)

    def to_string(self):
        print(self._data)

    def to_display(self):
        pass



