# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from .display_object import DisplayElement
from dash import dash_table


# For metrics with data per feature
class FeatureArrayElement(DisplayElement, requirements=['features']):
    def __init__(self, name, features=None):
        super().__init__(name)
        self.x = 0
        self._data["features"] = [{"name": "Measurement tag", "id": '0'}]
        self._data["row"] = []
        for i, feature in enumerate(features):
            self._data["features"].append({"name": feature, "id": str(i + 1)})

    def append(self, metric_data: list, tag: str):
        new_dict = {'0': tag}
        for i in range(len(metric_data)):
            new_dict[str(i + 1)] = metric_data[i]
        self._data["row"].append(new_dict)

    def to_string(self):
        print(self._data)

    def to_display(self):
        print("Columns: ", self._data["header"])
        print("data: ", self._data["row"])
        table = dash_table.DataTable(
            data=self._data["row"],
            columns=self._data["features"],
            style_table={'overflowX': 'auto'},
            export_headers='display',
            style_cell={'textAlign': 'center', 'padding-right': '10px', 'padding-left': '10px'},
            style_header={'background-color': 'rgb(240, 250, 255)'},
            merge_duplicate_headers=True)
        return table
