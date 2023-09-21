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
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from dash import html
from .server import app, dbUtils

logger = logging.getLogger(__name__)
unique_id = [1]
full = {}


def iconify(txt, icon, margin_right="10px", margin_left="0px"):
    style = {"margin-right": margin_right or 15, "margin-left": margin_left}
    return html.Div([html.I(className=icon, style=style), html.Span(txt, className='custom-icon')])


def dict_to_table(d, list_vertical=True):
    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th(x) for x in d.keys()])),
            html.Tbody(html.Tr([html.Td(process_cell(x, list_vertical)) for x in d.values()]))],
        bordered=True, hover=False, responsive=True, striped=True,
        style={"while-space": "normal", "padding-top": "12px", "padding-bottom": "12px"})


def to_str(v, max_len=None):
    if max_len is None:
        max_len = dbUtils._maxlen
    s = str(v)
    if len(s) > max_len:
        unique_id[0] += 1
        s = s[:max_len]
        full[s] = str(v)
        btn = dbc.Button("...", id={"type": str(v), "index": unique_id[0]},
                         outline=True, color="secondary", className="me-1", size="sm",
                         style={"margin-left": "5px", "width": "28px", "height": "20px",
                                "text-align": "center bottom", "line-height": "5px"})
        return html.Div([s, btn, html.Div(id='dummy_x')])
    return s


@app.callback(
    Output("dummy_x", "children"),
    Input({"type": dash.ALL, "index": dash.ALL}, "n_clicks"),
    State("dummy_x", "children"))
def show_full_text(x, c):
    if dash.callback_context.triggered_id is None:
        print("error: x is none ", x)
        return c

    else:
        full_text = dash.callback_context.triggered_id['type']
        return dbc.Offcanvas(
            html.P(full_text),
            id="offcanvas", title="Full Content", is_open=True)


def process_cell(v, list_vertical=True):
    if type(v) in (tuple, list):
        # return pp.pformat(v)
        if list_vertical:
            return [dbc.Row(dbc.Col(html.Div(to_str(x)))) for x in v]
        else:
            return to_str(v)

    if isinstance(v, dict):
        return dict_to_table(v, list_vertical)
    return to_str(v)
