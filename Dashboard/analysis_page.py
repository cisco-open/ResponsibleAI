import logging
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash
from server import app, redisUtil
import metric_view_functions as mvf
from dash import Input, Output, html, State
import dash_daq as daq
import numpy as np
import plotly.express as px
import pandas as pd

logger = logging.getLogger(__name__)


def get_analysis_page():
    redisUtil.request_available_analysis()
    options = redisUtil.get_available_analysis()
    result = html.Div([
        html.H4("Select The Analysis"),
        dcc.Interval(
            id='analysis-interval-component',
            interval=1 * 1000,  # in milliseconds
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
    is_time_update, is_button, is_value = mvf.analysis_update_cause(ctx, "analysis-")
    analysis_choices = redisUtil.get_available_analysis()
    if is_time_update:
        if redisUtil.has_update("analysis_update", reset=True):
            print("updating analysis data")
            analysis_display = [html.P(redisUtil.get_analysis(analysis_choice))]
            redisUtil._subscribers["analysis_update"] = False
    if is_button:
        if analysis_choice is None or analysis_choice == "":
            return analysis_choices, [html.P("Please select an analysis")]
        else:
            redisUtil.request_start_analysis(analysis_choice)
            return redisUtil.get_available_analysis(), [html.P("Starting analysis")]
    if is_value:
        analysis_display = [redisUtil.get_analysis(analysis_choice)]
    return analysis_choices, analysis_display

