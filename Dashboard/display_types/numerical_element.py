from .traceable_element import TraceableElement


# For single valued numeric metrics, like accuracy
class NumericalElement(TraceableElement):
    def __init__(self, name):
        super().__init__(name)
        self.x = 0
        self._data["x"] = []
        self._data["y"] = []
        self._data["tag"] = []
        self._data["text"] = []

    def append(self, metric_data, tag):
        self._data["x"].append(self.x)
        self.x += 1
        self._data["y"].append(metric_data)
        self._data["tag"].append(tag)
        self._data["text"].append("%.2f" % metric_data)

    def to_string(self):
        print(self._data)
