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


from .single_display_object import SingleDisplayElement
import plotly.graph_objs as go
import numpy as np
from dash import dcc


# For metrics which matrix return values
class MatrixElement(SingleDisplayElement):
    def __init__(self, name, row_names=None, col_names=None):
        super().__init__(name)
        self.x = 0
        self._data["features_y"] = None
        if row_names is not None:
            self._data["features_y"] = row_names.copy()
        self._data["features_x"] = None
        if col_names is not None:
            self._data["features_x"] = col_names.copy()
        if row_names is not None and col_names is not None:
            self._data["features_x"].insert(0, "")
        self._data["tag"] = []
        self._data["row"] = []
        self._data["matrices"] = []

    def append(self, metric_data, tag):
        self._data["matrices"].append(metric_data)
        self._data["tag"].append(tag)

    def to_string(self):
        print(self._data)

    def display_tag_num(self, num):
        data = self._data["matrices"][num]
        data = np.array(data, dtype=object)
        header = None
        if self._data["features_x"] is not None:
            header = self._data["features_x"]
        if self._data["features_y"] is not None:
            data = np.insert(data, 0, [self._data["features_y"]], axis=0)
        return [dcc.Graph(figure=go.Figure(data=[go.Table(header=dict(values=header), cells=dict(values=data))]))]

    def to_display(self):
        return self.display_tag_num(len(self._data["tag"]) - 1)
