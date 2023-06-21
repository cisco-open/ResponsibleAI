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
import plotly.graph_objs as go
from abc import ABCMeta
from dash import dcc


class TraceableElement(DisplayElement, metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__(name)
        self._data = {"x": [], "y": [], "tag": [], "text": []}
        self.x = 0
        self._settings = {"mode": 'lines+markers+text',
                          "orientation": 'v',
                          "showlegend": True,
                          "type": 'scatter',
                          "textposition": 'top center',
                          "tickmode": 'array',
                          "title_font_family": "Times New Roman",
                          "font_family": "Times New Roman",
                          "font_size": 14,
                          "font_color": "black",
                          "bgcolor": "Azure",
                          "bordercolor": "Black",
                          "borderwidth": 1}

    def _get_sc_data(self):
        return {
            'mode': self._settings["mode"], 'name': f"{self._name}",
            'orientation': self._settings["orientation"], 'showlegend': self._settings["showlegend"],
            'text': self._data["text"], 'x': self._data["x"], 'xaxis': 'x', 'y': self._data['y'], 'yaxis': 'y',
            'type': self._settings["type"], 'textposition': self._settings["textposition"],
            'hovertemplate': 'metric=' + self._name + '<br>x=%{x}<br>value=%{y}<br>text=%{text}<extra></extra>'
        }

    def _update_layout(self, fig, tickvals):
        fig.update_layout(
            xaxis=dict(tickmode=self._settings["tickmode"], tickvals=tickvals, ticktext=self._data["tag"]),
            legend=dict(title_font_family=self._settings["title_font_family"],
                        font=dict(family=self._settings["font_family"], size=self._settings["font_size"],
                                  color=self._settings["font_color"]),
                        bgcolor=self._settings["bgcolor"],
                        bordercolor=self._settings["bordercolor"],
                        borderwidth=self._settings["borderwidth"]))

    def to_display(self):
        sc_data = self._get_sc_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(**sc_data))
        fig.update_traces(textposition="top center")
        self._update_layout(fig, self._data["x"])
        return [dcc.Graph(figure=fig)]

    def add_trace_to(self, fig):
        sc_data = self._get_sc_data()
        fig.add_trace(go.Scatter(**sc_data))
        fig.update_traces(textposition="top center")
        self._update_layout(fig, self._data["x"])
        return fig
