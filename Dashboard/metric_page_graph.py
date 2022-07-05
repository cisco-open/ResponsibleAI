import logging
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, html, State
from server import app, redisUtil
from dash import dcc
import plotly.express as px
import plotly.graph_objs as go
from display_types import NumericalElement, BooleanElement
logger = logging.getLogger(__name__)


def add_trace_to(fig, group, metric):
    d = {"x": [], "y": [], "tag": [], "metric": [], "text": []}
    dataset = redisUtil.get_current_dataset()

    metric_values = redisUtil.get_metric_values()
    metric_type = redisUtil.get_metric_info()
    type = metric_type[group][metric]["type"]
    display_obj = None
    if type == "numeric":
        display_obj = NumericalElement(metric)
    elif type == "boolean":
        display_obj = BooleanElement(metric)
    for i, data in enumerate(metric_values):
        data = data[dataset]
        print("Metric value: ", metric, " ", data[group][metric])
        display_obj.append(data[group][metric], data["metadata"]["tag"])
    display_obj.add_trace_to(fig)
    return


def get_selectors():
    groups = []
    for g in redisUtil.get_metric_info():
        groups.append(g)

    return html.Div(
        dbc.Form([
            dbc.Label("select metric group", html_for="select_group"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(groups, id='select_group', value=groups[0] if groups else None, persistence=True,
                                 persistence_type='session', placeholder="Select a metric group", ),
                    html.P(""),
                    dbc.Label("select metric", html_for="select_metric_cnt"),
                    dcc.Dropdown([], id='select_metric_dd', value=[], placeholder="Select a metric", persistence=True,
                                 persistence_type='session'),
                ], style={"width": "70%"}),
                dbc.Col([
                    dbc.Button("Reset Graph", id="reset_graph", style={"margin-left": "20%"}, color="secondary")
                ], style={"width": "20%"})
            ])],
            style={"background-color": "rgb(240,250,255)", "width": "100%  ", "border": "solid",
                  "border-color": "silver", "border-radius": "5px", "padding": "50px"}
        ),
        style={"margin": "2px", "margin-bottom": "20px",}
    )


def get_graph():
    d = {"x": [], "value": [], "tag": [], "metric": []}
    fig = px.line(pd.DataFrame(d), x="x", y="value", color="metric", markers="True")
    return html.Div(
        html.Div(id='graph_cnt', children=[dcc.Graph(figure=fig, id='metric_graph')], style={"margin": "1"}),
        style={
            "border-width": "thin",
            "border-color": "LightGray",
            "border-style": "solid",
            "border-radius": "3px",
            "margin": "1"})


def get_metric_page_graph():
    return html.Div([
        dcc.Store(id='legend_data', storage_type='local'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0),
        html.Div(html.Div([get_selectors()])),
        get_graph()])


@app.callback(
    Output('select_metric_dd', 'options'),
    Input('select_group', 'value'))
def update_metrics(value):
    print("updating metrics")
    if not value:
        logger.info("no value for update")
        return []
        # return dcc.Dropdown([], id='select_metrics', persistence=True, persistence_type='session')
    metrics = []
    for m in redisUtil.get_metric_info()[value]:
        if m == "meta":
            continue
        if redisUtil.get_metric_info()[value][m]["type"] in ["numeric", "boolean"]:
            metrics.append(m)
    return metrics
    # return dcc.Dropdown( metrics,  id='select_metrics',persistence=True, persistence_type='session')


def create_options_children(options):
    res = []
    for item in options:
        res.append(dbc.ListGroupItem(item))
    return res


@app.callback(
    Output('legend_data', 'data'),
    Input('select_metric_dd', 'value'),
    Input('reset_graph', "n_clicks"),
    State('select_group', 'value'),
    State('legend_data', 'data')
)
def update_options(metric, clk, group, options):
    print("metric: ", metric, ", group: ", group, ", options: ", options)
    ctx = dash.callback_context
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'reset_graph.n_clicks':
        return []
    if metric is None or group is None:
        return options  # [] set to options to retain settings
    item = group + "," + metric
    if item not in options:
        options.append(item)
    print("options: ", options)
    return options


@app.callback(
    Output('metric_graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('legend_data', 'data'),
    State('metric_graph', 'figure')
)
def update_graph(n, options, old):
    ctx = dash.callback_context
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'interval-component.n_intervals':
        if redisUtil.has_update("metric_graph", reset=True):
            logger.info("new data")
            redisUtil._subscribers["metric_graph"] = False
        else:
            return old

    fig = go.Figure()
    if len(options) == 0:
        return fig
    print("options: ", options)
    for item in options:
        print("split: ", item.split(','))
        k, v = item.split(',')
        print(k, v)
        print('---------------------------')
        add_trace_to(fig, k, v)
    return fig
