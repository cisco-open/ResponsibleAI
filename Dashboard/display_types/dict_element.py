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
            i += 1
        self._data["row"].append(new_dict)

    def to_string(self):
        print(self._data)

    def to_display(self):
        header = [{"name": i, "id": str(num)} for num, i in enumerate(self._data["features"])]

        header.insert(0, {"name": ["RAI Tag"], "id": '-1'})
        tagged_data = [{str(k): v for k, v in val.items()} for val in self._data["row"].copy()]
        for i, row in enumerate(tagged_data):
            row['-1'] = self._data["tag"][i]

        print("Columns: ", header)
        print("data: ", tagged_data)

        table = dash_table.DataTable(
            data=tagged_data,
            columns=header,
            style_table={'overflowX': 'auto'},
            export_headers='display',
            style_cell={'textAlign': 'center', 'padding-right': '10px', 'padding-left': '10px'},
            style_header={'background-color': 'rgb(240, 250, 255)'},
            merge_duplicate_headers=True)
        return table
