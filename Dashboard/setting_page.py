import logging
import dash_bootstrap_components as dbc
from dash import Input, Output, html
from server import app, redisUtil
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
                      dbc.Input(id="input_precision", type="number", min=0, max=6, step=1, value=redisUtil._precision),
                      ], id="styled-numeric-input", className="d-grid gap-2"),
            html.Div(
                [dbc.FormText("Maximum text length", style={"margin-top": "20px"}),
                 dbc.Input(id="input_maxlen", type="number", min=1, max=500, step=1, value=redisUtil._maxlen)],
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
        redisUtil._maxlen = maxlen
    if precision:
        redisUtil.reformat(precision)
    return []
