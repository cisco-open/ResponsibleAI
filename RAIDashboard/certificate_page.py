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
from dash import Input, Output, dcc, html
from .server import app, dbUtils

logger = logging.getLogger(__name__)


def get_form():
    ops = []
    dataset = dbUtils.get_current_dataset()
    values = dbUtils.get_metric_values()

    for i, m in enumerate(values):
        ops.append({"label": m[dataset]["metadata"]["date"] + " - " + m[dataset]["metadata"]["tag"], "value": i})

    dropdown = html.Div(
        [
            dbc.Label("Select Measurement", html_for="dropdown"),
            dcc.Dropdown(id="measurement_selector_cert", options=ops[::-1], value=len(values) - 1)
        ],
        className="mb-3",
    )
    return dbc.Form([dropdown])


def get_cert_name(cert_id):
    return dbUtils.get_certificate_info()[cert_id]['display_name']


def generate_cert_table(id, show_explanation=True):
    rows = []
    if len(dbUtils.get_certificate_values()) > 0:
        for k, v in dbUtils.get_certificate_values().items():
            if v['value']:
                status = html.Div([
                    "Passed ", html.I(className="fa-solid fa-check",
                                      style={"width": "30px", "height": "30px", "margin-left": "10px", "color": "green"})
                ])
            else:
                status = html.Div([
                    "Failed", html.I(className="fa-solid fa-xmark",
                                     style={"width": "30px", "height": "30px", "margin-left": "25px", "color": "red"})
                ])
            rows.append(html.Tr(
                [html.Td(get_cert_name(k).title()), html.Td(v['explanation']), html.Td(status)] if show_explanation else
                [html.Td(get_cert_name(k).title()), html.Td(status)]
            ))
    return dbc.Table(
        children=[
            html.Thead(
                html.Tr([html.Th("Certificate"), html.Th("Explanation"), html.Th("Status")] if show_explanation else
                        [html.Th("Certificate"), html.Th("Status")])
            ),
            html.Tbody(rows)
        ], striped=True
    )


def get_certificate_page():
    return html.Div([
        html.P(""),
        html.P(""),
        html.P(""),
        html.P(""),
        html.P(""),
        html.Div(
            html.Div(get_form(), style={"margin": "20px"}),
            style={"background-color": "Azure",
                   "border-width": "thin",
                   "border-color": "Blue",
                   "border-style": "solid",
                   "border-radius": "10px",
                   }
        ),
        html.Hr(),
        html.Div(
            html.Div(id="certificate_pane", style={"margin": "30px"}),
            style={"margin": "5",
                   "border-width": "thin",
                   "border-color": "Blue",
                   "border-style": "solid",
                   "border-radius": "10px", })
    ])


@app.callback(
    Output('certificate_pane', 'children'),
    Input('measurement_selector_cert', 'value'),
)
def update_metrics(value):
    return generate_cert_table(value)
