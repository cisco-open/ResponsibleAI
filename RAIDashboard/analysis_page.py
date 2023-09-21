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
from dash import dcc
import dash
from .server import app, dbUtils
from dash import Input, Output, html, State
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)


def get_analysis_page():
    dbUtils.request_available_analysis()
    options = dbUtils.get_available_analysis()
    result = html.Div([
        html.H4("Select The Analysis"),
        dcc.Interval(
            id='analysis-interval-component',
            interval=1 * 1500,  # in milliseconds
            n_intervals=0),
        dcc.Dropdown(
            id="analysis_selector",
            options=options,
            value=None,
            persistence=True),
        html.Button("Run Analysis", id="run_analysis_button", style={"margin-top": "20px"}),
        html.Div([], id="analysis_display", style={"margin-top": "20px"})
    ], style={})
    return result


@app.callback(
    Output('analysis_selector', 'options'),
    Output('analysis_display', 'children'),
    Input('analysis-interval-component', 'n_intervals'),
    Input('run_analysis_button', 'n_clicks'),
    Input('analysis_selector', 'value'),
    State('analysis_selector', 'options'),
    State('analysis_display', 'children'),
)
def get_analysis_updates(timer, btn, analysis_choice, analysis_choices, analysis_display):
    ctx = dash.callback_context
    is_time_update = any('analysis-interval-component.n_intervals' in i['prop_id'] for i in ctx.triggered)
    is_button = any('run_analysis_button.n_clicks' in i['prop_id'] for i in ctx.triggered)
    is_value = any('analysis_selector.value' == i['prop_id'] for i in ctx.triggered)
    should_update = False
    if analysis_choices != dbUtils.get_available_analysis():
        analysis_choices = dbUtils.get_available_analysis()
        should_update = True
    if is_time_update and analysis_choice is not None:
        if dbUtils.has_analysis_update(analysis_choice, reset=True):
            print("Analysis update: ")
            analysis_display = [dbUtils.get_analysis(analysis_choice),
                                html.P(analysis_choice, style={"display": "none"})]
            return analysis_choices, analysis_display
    if is_button:
        if analysis_choice is None or analysis_choice == "":
            return analysis_choices, [html.P("Please select an analysis")]
        else:
            dbUtils.request_start_analysis(analysis_choice)
            return dbUtils.get_available_analysis(), [html.P("Requesting Analysis..")]

    # Extra condition was added because dash would not always update when changing to/from a large analysis
    if is_value or (
            len(analysis_display) > 1 and analysis_choice != analysis_display[1].get("props", {}).get("children", {})
    ):
        analysis_display = [dbUtils.get_analysis(analysis_choice),
                            html.P(analysis_choice, style={"display": "none"})]
        should_update = True

    if not should_update:
        raise PreventUpdate

    return analysis_choices, analysis_display
