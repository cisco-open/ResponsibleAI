import logging
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, html, State
from server import app, redisUtil
from dash import dcc
import plotly.express as px
from display_types import NumericalElement, FeatureArrayElement, BooleanElement, MatrixElement, DictElement
logger = logging.getLogger(__name__)


def get_display_data(group, metric):
    d = {"x": [], "y": [], "tag": [], "metric": [], "text": []}
    dataset = redisUtil.get_current_dataset()
    # TODO: Make a list of which metric_values instances use the dataset
    metric_values = redisUtil.get_metric_values()
    metric_type = redisUtil.get_metric_info()
    type = metric_type[group][metric]["type"]
    display_obj = None
    if type == "numeric":
        display_obj = NumericalElement(metric)
    elif type == "feature-array":
        display_obj = FeatureArrayElement(metric, redisUtil.get_project_info()["features"])
    elif type == "boolean":
        display_obj = BooleanElement(metric)
    elif type == "matrix":
        display_obj = MatrixElement(metric)
    elif type == "dict":
        display_obj = DictElement(metric)
    else:
        assert("Metric type " + type + " must be one of (numeric, vector-array, bool, matrix, dict)")
    for i, data in enumerate(metric_values):
        data = data[dataset]
        print("Value: ", data[group][metric])
        display_obj.append(data[group][metric], data["metadata"]["tag"])
    return display_obj, display_obj.requires_tag_chooser


def get_selectors():
    groups = []
    for g in redisUtil.get_metric_info():
        groups.append(g)

    return html.Div(
        dbc.Form([
            dbc.Label("select metric group", html_for="select_group"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(groups, id='indiv_select_group', value=groups[0] if groups else None, persistence=True,
                                 persistence_type='session', placeholder="Select a metric group"),
                    html.P(""),
                    dbc.Label("select metric", html_for="select_metric_cnt"),
                    dcc.Dropdown([], id='indiv_select_metric_dd', value=None, placeholder="Select a metric",
                                 persistence=True, persistence_type='session'),
                ], style={"width": "70%"}),
                dbc.Col([
                    dbc.Button("Reset Graph", id="indiv_reset_graph", style={"margin-left": "20%"}, color="secondary")
                ], style={"width": "20%"}),
            ]),
            dbc.Row([dbc.Col([
                html.P(""),
                dbc.Label("Select Tag", html_for="select_metric_tag"),
                dcc.Dropdown([], id='indiv_select_metric_tag', value=None, placeholder="Select a tag",
                     persistence=True, persistence_type='session')], id="select_metric_tag_col", style={"display": 'none'})],
                id="indiv_tag_selector_row")],
            style={"background-color": "rgb(240,250,255)", "width": "100%  ", "border": "solid",
                  "border-color": "silver", "border-radius": "5px", "padding": "50px"}
        ),
        style={"margin": "2px", "margin-bottom": "20px"}
    )


def get_graph():
    d = {"x": [], "value": [], "tag": [], "metric": []}
    fig = px.line(pd.DataFrame(d), x="x", y="value", color="metric", markers="True")
    return html.Div(
        html.Div(id='indiv_graph_cnt', children=[], style={"margin": "1"}),
        style={
            "border-width": "thin",
            "border-color": "LightGray",
            "border-style": "solid",
            "border-radius": "3px",
            "margin": "1"})


def get_single_metric_display():
    return html.Div([
        dcc.Store(id='indiv_legend_data', storage_type='local'),
        dcc.Interval(
            id='indiv-interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0),
        html.Div(html.Div([get_selectors()])),
        get_graph()])


@app.callback(
    Output('indiv_select_metric_dd', 'options'),
    Input('indiv_select_group', 'value'))
def update_metrics(value):
    if not value:
        logger.info("no value for update")
        return []
        # return dcc.Dropdown([], id='select_metrics', persistence=True, persistence_type='session')
    metrics = []
    for m in redisUtil.get_metric_info()[value]:
        if m == "meta":
            continue
        if redisUtil.get_metric_info()[value][m]["type"] in ["numeric", "feature-array", "boolean", "matrix", "dict"]:
            metrics.append(m)
    return metrics
    # return dcc.Dropdown( metrics,  id='select_metrics',persistence=True, persistence_type='session')


@app.callback(
    Output('indiv_legend_data', 'data'),
    Input('indiv_select_metric_dd', 'value'),
    Input('indiv_reset_graph', "n_clicks"),
    State('indiv_select_group', 'value'),
    State('indiv_legend_data', 'data')
)
def update_options(metric, btn, group, options):
    ctx = dash.callback_context
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'indiv_reset_graph.n_clicks':
        return None
    if metric is None or group is None:
        return options  # None set to options to retain settings
    return group + "," + metric


@app.callback(
    Output('indiv_graph_cnt', 'children'),
    Output('select_metric_tag_col', 'style'),
    Output('indiv_select_metric_tag', 'options'),
    Output('indiv_select_metric_tag', 'value'),
    Input('indiv-interval-component', 'n_intervals'),
    Input('indiv_legend_data', 'data'),
    Input('indiv_select_metric_tag', 'value'),
    State('indiv_graph_cnt', 'children'),
    State('select_metric_tag_col', 'style'),
    State('indiv_select_metric_tag', 'options'),
    State('indiv_select_metric_tag', 'value')
)
def update_graph(n, options, tag_selection, old_graph, old_style, old_children, old_value):
    ctx = dash.callback_context
    style = {"display": 'none'}
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'indiv-interval-component.n_intervals':
        if redisUtil.has_update("metric_graph", reset=True):
            logger.info("new data")
            redisUtil._subscribers["metric_graph"] = False
        else:
            return old_graph, old_style, old_children, old_value
    elif 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'indiv_select_metric_tag.value':
        print("Different value selected")
        k, v = options.split(',')
        display_obj, _ = get_display_data(k, v)
        tags = display_obj.get_tags()
        return display_obj.display_tag_num(tags.index(tag_selection)), old_style, old_children, tag_selection
    elif options == "" or options == None:
        return [], style, old_children, old_value
    k, v = options.split(',')
    print(k, v)
    print('---------------------------')
    display_obj, needs_chooser = get_display_data(k, v)
    children = []
    selection = None
    if needs_chooser:
        style = {"display": 'block'}
        children = display_obj.get_tags()
        selection = children[-1]
    return display_obj.to_display(), style, children, selection
