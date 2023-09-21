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


import logging
import dash_bootstrap_components as dbc
from dash import Input, Output, html
from .server import app, dbUtils
import urllib
logger = logging.getLogger(__name__)


def get_setting_page():
    qs = urllib.parse.urlencode({"g": "metadata", "m": "date"})
    return html.Div([
        dbc.Form([
            html.H4("Display Settings"),
            html.Div(id="dummy_setting"),
            html.Hr(),
            html.Div([dbc.FormText("Precision for floating points data"),
                      dbc.Input(id="input_precision", type="number", min=0, max=6, step=1, value=dbUtils._precision),
                      ], id="styled-numeric-input", className="d-grid gap-2"),
            html.Div(
                [dbc.FormText("Maximum text length", style={"margin-top": "20px"}),
                 dbc.Input(id="input_maxlen", type="number", min=1, max=500, step=1, value=dbUtils._maxlen)],
                className="d-grid gap-2", id="styled-numeric-input"),
            html.Div(
                dbc.Button("Apply", id="apply_setting", href="/single_metric_info/?" + qs, style={"margin-top": "30px"},
                           color="secondary"), className="d-grid gap-2")
        ], style={"width": "400px", "border": "solid", "border-color": "silver", "border-radius": "5px",
                  "padding": "50px"}
        ),
        html.Div(id="setting_div")
    ])


@app.callback(
    Output("dummy_setting", "children"),
    [
        Input("input_maxlen", "value"),
        Input("input_precision", "value"),
    ],
)
def on_form_change(maxlen, precision):
    if maxlen is not None:
        dbUtils._maxlen = maxlen
    if precision:
        dbUtils.reformat(precision)
    return []
