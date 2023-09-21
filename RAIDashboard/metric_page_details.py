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
from dash import dcc
from .utils import process_cell
logger = logging.getLogger(__name__)


def get_accordion(id):
    items = []
    dataset = dbUtils.get_current_dataset()
    values = dbUtils.get_metric_values()[id][dataset]
    metric_info = dbUtils.get_metric_info()

    for group in values:
        rows = []
        for k, v in values[group].items():
            if isinstance(v, dict):
                vs = []
                for ki, vi in v.items():
                    vs.append(html.Div(process_cell({ki: vi}), style={"max-width": "1200px"}))
            else:
                vs = process_cell(v, list_vertical=isinstance(v, dict))

            qs = urllib.parse.urlencode({"g": group, "m": k})
            ico = html.I(n_clicks=0, className='fa-solid fa-info')
            btn = dbc.Button(
                html.Span([ico]), href="/single_metric_info/?" + qs, style={"width": "3px", "margin-right": "5px"},
                outline=True, color="light", className="me-1")
            rows.append(html.Tr([html.Td(html.Div([btn, k])), html.Td(vs)]))

        detail = dbc.Table(
            children=[html.Thead(html.Tr([html.Th("Metric Name"), html.Th("Metric Value")])), html.Tbody(rows)],
            bordered=True, striped=True, responsive=True, size='sm')
        items.append(
            dbc.AccordionItem(children=detail,
                              title=metric_info[group]["meta"]["display_name"],
                              item_id=group)
        )
    return dbc.Accordion(items, active_item=items[0].item_id, flush=True)


def get_form():
    ops = []
    dataset = dbUtils.get_current_dataset()
    values = dbUtils.get_metric_values()
    for i, m in enumerate(values):
        m = m[dataset]
        ops.append({"label": m["metadata"]["date"] + " - " + m["metadata"]["tag"], "value": i})

    dropdown = html.Div([
        dbc.Label("Select Measurement", html_for="dropdown"),
        dcc.Dropdown(id="measurement_selector", options=ops[::-1], value=len(values) - 1)],
        className="mb-3")

    return dbc.Form([dropdown])


def get_metric_page_details():
    return html.Div([
        html.P(""),
        html.P(""),
        html.P(""),
        html.P(""),
        html.P(""),
        html.Div(
            html.Div(get_form(), style={"margin": "20px"}),
            style={"background-color": "rgb(198,216,233)",
                   "border-width": "thin",
                   "border-color": "silver",
                   "border-style": "solid",
                   "border-radius": "10px",
                   "padding": "10px"}
        ),
        html.Hr(),
        html.Div(html.Div(id="measure_accordion"))])


@app.callback(
    Output('measure_accordion', 'children'),
    Input('measurement_selector', 'value'),
)
def update_metrics(value):
    return get_accordion(value)
