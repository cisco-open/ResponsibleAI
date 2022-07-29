from .display_object import DisplayElement
from dash import dash_table


# For metrics which return arrays unrelated to features
class VectorElement(DisplayElement):
    def __init__(self, name):
        super().__init__(name)
        self.x = 0
        self._data["row"] = []
        self._data["header"] = []

    def append(self, metric_data, tag):
        if len(self._data["header"]) == 0:
            print("Setting header")
            self._data["header"] = [{"name": ["Measurement Tag"], "id": 0}]
            for i in range(len(metric_data)):
                self._data["header"].append({"name": [''], "id": i+1})
        dict_result = {0: tag}
        for i in range(len(metric_data)):
            dict_result[i + 1] = metric_data[i]
        self._data["row"].append(dict_result)

    def to_string(self):
        print(self._data)

    def to_display(self):
        print("Columns: ", self._data["header"])
        print("data: ", self._data["row"])
        table = dash_table.DataTable(
            data=self._data["row"],
            columns=self._data["header"],
            style_table={'overflowX': 'auto'},
            export_headers='display',
            style_cell={'textAlign': 'center', 'padding-right': '10px', 'padding-left': '10px'},
            style_header={'background-color': 'rgb(240, 250, 255)'},
            merge_duplicate_headers=True)
        return table
