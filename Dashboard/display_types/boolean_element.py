from .traceable_element import TraceableElement


# For single valued boolean metrics
class BooleanElement(TraceableElement):
    def __init__(self, name):
        super().__init__(name)
        self.x = 0
        self._data["x"] = []
        self._data["y"] = []
        self._data["tag"] = []
        self._data["text"] = []

    def to_string(self):
        print(self._data)

    def append(self, metric_data, tag):
        assert isinstance(metric_data, bool), "metric_data must be of type bool not " + type(metric_data)
        val = 0
        if metric_data:
            val += 1
        self._data["x"].append(self.x)
        self.x += 1
        self._data["y"].append(val)
        self._data["tag"].append(tag)
        self._data["text"].append(str(metric_data))
