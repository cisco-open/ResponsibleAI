from .display_object import DisplayElement
import plotly.graph_objs as go
from dash import Dash, dash_table


# For metrics which consist of one or more layers of dictionaries
class DictElement(DisplayElement):
    def __init__(self, name):
        super().__init__(name)
        self._data["features"] = None
        self._data["row"] = []
        self._data["tag"] = []
        self.x = 0

    def _dfs(self, my_dict, prev, result):
        for key in my_dict.keys():
            old_prev = prev.copy()
            prev.append(key)
            if isinstance(my_dict[key], dict):
                self._dfs(my_dict[key], prev, result)
            else:
                result.append(prev)
            prev = old_prev

    def append(self, metric_data, tag):
        if self._data["features"] is None:
            result = []
            self._dfs(metric_data, [], result)
            self._data["features"] = result

        self._data["tag"].append(tag)
        i = 0
        new_dict = {}
        for features in self._data["features"]:
            res = metric_data
            for val in features:
                res = res[val]
            new_dict[i] = res
            i+=1
        self._data["row"].append(new_dict)

    def to_string(self):
        print(self._data)

    def to_display(self):
        header = [{"name": i, "id": num} for num, i in enumerate(self._data["features"])]

        header.insert(0, {"name": ["RAI Tag"], "id": -1})
        tagged_data = self._data["row"].copy()
        for i, row in enumerate(tagged_data):
            row[-1] = self._data["tag"][i]

        table = dash_table.DataTable(
            data=tagged_data,
            columns=header,
            style_table={'overflowX': 'auto'},
            export_headers='display',
            style_cell={'textAlign': 'center', 'padding-right': '10px', 'padding-left': '10px'},
            style_header={'background-color': 'rgb(240, 250, 255)'},
            merge_duplicate_headers=True)
        return table
