from .display_object import DisplayElement
import plotly.graph_objs as go


# For metrics with data per feature
class FeatureArrayElement(DisplayElement):
    def __init__(self, name, features):
        super().__init__(name)
        self.x = 0
        self._data["features"] = features.copy()
        self._data["features"].insert(0, "Tag")
        self._data["row"] = []
        for i in range(len(self._data["features"])):
            self._data["row"].append([])
        print("starting data: ", self._data["row"])
        print("starting features: ", self._data["features"])

    def append(self, metric_data, tag):
        self._data["row"][0].append(tag)
        for i in range(len(metric_data)):
            self._data["row"][i+1].append(metric_data[i])

    def to_string(self):
        print(self._data)

    def to_display(self):
        return go.Figure(data=[go.Table(header=dict(values=self._data["features"]),
                                        cells=dict(values=self._data["row"]))])
