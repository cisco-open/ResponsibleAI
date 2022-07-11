import logging
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, html, State
from server import app, redisUtil
from dash import dcc, ALL
import plotly.express as px
from display_types import NumericalElement, FeatureArrayElement, BooleanElement, MatrixElement, DictElement
logger = logging.getLogger(__name__)

requirements = ["numeric", "feature-array", "boolean", "matrix", "dict"]


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


def get_nonempty_groups(requirements):
    metric_info = redisUtil.get_metric_info()
    valid_groups = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and metric_info[group][metric]["type"] in requirements:
                valid_groups.append(group)
                break
    return valid_groups


def get_valid_metrics(group):
    metric_info = redisUtil.get_metric_info()
    valid_metrics = []
    for metric in metric_info[group]:
        if "type" in metric_info[group][metric] and metric_info[group][metric]["type"] in requirements:
            valid_metrics.append(metric)
    return valid_metrics


def get_checklist():
    groups = get_nonempty_groups(requirements)
    return html.Div([
            html.Details([
                html.Summary([html.P([group], style={"display": "inline-block", "margin-bottom": "0px"})]),
                dcc.RadioItems(
                    id={"type": "indiv-child-checkbox", "group": group},
                    options=[
                        {"label": " " + i, "value": i} for i in get_valid_metrics(group)
                    ],
                    value=[],
                    labelStyle={"display": "block"},
                    style={"padding-left": "40px"}
                )]) for group in groups], style={"margin-left": "35%"})


def get_search_options():
    metric_info = redisUtil.get_metric_info()
    valid_searches = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and metric_info[group][metric]["type"] in requirements:
                valid_searches.append(metric + " | " + group)
    return valid_searches


def get_selectors():
    groups = []
    for g in redisUtil.get_metric_info():
        groups.append(g)

    tab_style = {
        'borderBottom': '1px solid #d6d6d6',
        'padding': '6px',
        'fontWeight': 'bold'
    }
    tab_selected_style = {
        'borderTop': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': '#119DFF',
        'color': 'white',
        'padding': '6px'
    }

    return html.Div(
        dbc.Form([
            dcc.Tabs([
                dcc.Tab(label='Metric Selector', children=[
                    dbc.Row([
                        dbc.Col([
                            get_checklist(),
                        ], style={"position": "relative"}),
                        dbc.Col([
                            dbc.Button("Reset Graph", id="indiv_reset_graph", color="secondary",
                                       style={"position": "absolute", "bottom": "0"}),
                        ], style={"position": "relative"}),
                    ], style={"width": "100%", "margin-top": "20px"}),
                ], selected_style=tab_selected_style, style=tab_style),
                dcc.Tab(label='Metric Search', children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(get_search_options(), id='indiv_metric_search', value=None,
                                         placeholder="Search Metrics"),
                        ], style={"position": "relative"}),
                        dbc.Col([
                            dbc.Button("Reset Graph", id="indiv_reset_graph", color="secondary"),
                        ], style={"position": "relative"})
                    ], style={"width": "100%", "margin-top": "20px"}),
                ], selected_style=tab_selected_style, style=tab_style),
            ]),
            dbc.Row([dbc.Col([
                html.P(""),
                dbc.Label("Select Tag", html_for="select_metric_tag"),
                dcc.Dropdown([], id='indiv_select_metric_tag', value=None, placeholder="Select a tag",
                             persistence=True, persistence_type='session')], id="indiv_select_metric_tag_col",
                style={"display": 'none'})],
                id="indiv_tag_selector_row")
        ], style={"background-color": "rgb(240,250,255)", "width": "100%", "border": "solid",
                  "border-color": "silver", "border-radius": "5px", "padding": "10px 50px 10px 50px"}),
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


def get_group_from_ctx(ctx):
    search_string = "\"group\":\""
    idx = ctx.index(search_string) + len(search_string)
    idx_2 = ctx.index("\"", idx)
    return ctx[idx:idx_2]


@app.callback(
    Output('indiv_legend_data', 'data'),
    Output({'type': 'indiv-child-checkbox', 'group': ALL}, 'value'),
    Output('indiv_metric_search', 'value'),
    Input({'type': 'indiv-child-checkbox', 'group': ALL}, 'value'),
    Input('indiv_reset_graph', "n_clicks"),
    Input('indiv_metric_search', 'value'),
    State({'type': 'indiv-child-checkbox', 'group': ALL}, 'options'),
    State({'type': 'indiv-child-checkbox', 'group': ALL}, 'value'),
    State({'type': 'indiv-child-checkbox', 'group': ALL}, 'id'),
    State('indiv_legend_data', 'data'),
    prevent_initial_call=True
)
def group_click(c_selected, reset_button, metric_search, c_options, c_val, c_ids, options):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    print("CONTEXT: ", ctx)
    ids = [c_ids[i]['group'] for i in range(len(c_ids))]
    if ctx == 'indiv_reset_graph.n_clicks':
        to_c_val = [[] for _ in range(len(c_val))]
        return "", to_c_val, None
    if ctx == 'indiv_metric_search.value':
        if metric_search is None:
            return options, c_val, metric_search
        else:
            vals = metric_search.split(" | ")
            metric_name = vals[1] + ',' + vals[0]
            metric = vals[0]
            group = vals[1]
            parent_index = ids.index(group)
            print("metric_name: ", metric_name)
            if metric_name != options:
                options = metric_name
                c_val = [[] for _ in range(len(c_val))]
                c_val[parent_index] = metric
            return options, c_val, metric_search
    group = get_group_from_ctx(ctx)
    parent_index = ids.index(group)
    if "\"type\":\"indiv-child-checkbox" in ctx:
        metric = c_val[parent_index]
        options = group + "," + c_val[parent_index]
        c_val = [[] for _ in range(len(c_val))]
        c_val[parent_index] = metric
        print("options: ", options)
        return options, c_val, None
    return options, c_val, None


@app.callback(
    Output('indiv_graph_cnt', 'children'),
    Output('indiv_select_metric_tag_col', 'style'),
    Output('indiv_select_metric_tag', 'options'),
    Output('indiv_select_metric_tag', 'value'),
    Input('indiv-interval-component', 'n_intervals'),
    Input('indiv_legend_data', 'data'),
    Input('indiv_select_metric_tag', 'value'),
    State('indiv_graph_cnt', 'children'),
    State('indiv_select_metric_tag_col', 'style'),
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
    print("this ran")
    if needs_chooser:
        style = {"display": 'block'}
        children = display_obj.get_tags()
        selection = children[-1]
    return display_obj.to_display(), style, children, selection