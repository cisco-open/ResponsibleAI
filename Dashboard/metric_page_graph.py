import logging
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, html, State
from server import app, redisUtil
from dash import dcc, MATCH, ALL, ALLSMALLER
import plotly.express as px
import plotly.graph_objs as go
from display_types import NumericalElement, BooleanElement
logger = logging.getLogger(__name__)

requirements = ["numeric", "boolean"]


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
        # print("Metric value: ", metric, " ", data[group][metric])
        display_obj.append(data[group][metric], data["metadata"]["tag"])
    display_obj.add_trace_to(fig)
    return


def get_nonempty_groups(requirements):
    metric_info = redisUtil.get_metric_info()
    valid_groups = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and metric_info[group][metric]["type"] in requirements:
                valid_groups.append(group)
                break
    return valid_groups


def get_search_options():
    metric_info = redisUtil.get_metric_info()
    valid_searches = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and metric_info[group][metric]["type"] in requirements:
                valid_searches.append(metric + " | " + group)
    return valid_searches


def get_valid_metrics(group):
    metric_info = redisUtil.get_metric_info()
    valid_metrics = []
    for metric in metric_info[group]:
        if "type" in metric_info[group][metric] and metric_info[group][metric]["type"] in requirements:
            valid_metrics.append(metric)
    return valid_metrics


def get_selection_update(group: str, selected: list, options: list):
    options = [item for item in options if not item.startswith(group+",")]
    for val in selected:
        options.append(group + "," + val)
    return options


def get_checklist():
    groups = get_nonempty_groups(requirements)
    return html.Div([
            html.Details([
                html.Summary([dcc.Checklist(
                    id={"type": "group-checkbox", "group": group},
                    options=[{"label": " " + group, "value": group}],
                    value=[],
                    labelStyle={"display": "inline-block"},
                    style={"display": "inline-block"},
                )]),
                dcc.Checklist(
                    id={"type": "child-checkbox", "group": group},
                    options=[{"label": " " + i, "value": i} for i in get_valid_metrics(group)],
                    value=[],
                    labelStyle={"display": "block"},
                    style={"padding-left": "40px"}
                )]) for group in groups])


def get_default_metric_options(groups):
    if len(groups) == 0:
        return []
    result = []
    for m in redisUtil.get_metric_info()[groups[0]]:
        if m == "meta":
            continue
        if redisUtil.get_metric_info()[groups[0]][m]["type"] in requirements:
            result.append(m)
    return result


def get_selectors():
    groups = []
    for g in redisUtil.get_metric_info():
        groups.append(g)

    return html.Div(
        dbc.Form([
            dbc.Row([
                dbc.Col([
                    get_checklist()
                ], style={"width": "70%"}),
                dbc.Col([
                    dcc.Dropdown(get_search_options(), id='metric_search', value=None, placeholder="Search Metrics"),
                    html.Br(),
                    dbc.Button("Reset Graph", id="reset_graph", color="secondary"),
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
        html.Div(id='graph_cnt', children=[], style={"margin": "1"}),
        style={
            "border-width": "thin",
            "border-color": "LightGray",
            "border-style": "solid",
            "border-radius": "3px",
            "margin": "1"})


def get_metric_page_graph():
    return html.Div([
        dcc.Store(id='legend_data', storage_type='local'),
        dcc.Store(id={'type': 'legend_data', 'group': ''}, storage_type='local'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0),
        html.Div(html.Div([get_selectors()])),
        get_graph()])


@app.callback(
    Output('select_metric_dd', 'options'),
    Output('display_group', 'style'),
    Input('select_group', 'value'))
def update_metrics(value):
    print("updating metrics")
    display_style = {"margin-left": "20%", "display": 'none'}
    if not value:
        logger.info("no value for update")
        return [], display_style
        # return dcc.Dropdown([], id='select_metrics', persistence=True, persistence_type='session')
    metrics = []
    for m in redisUtil.get_metric_info()[value]:
        if m == "meta":
            continue
        if redisUtil.get_metric_info()[value][m]["type"] in requirements:
            metrics.append(m)
    display_style['display'] = 'block'
    return metrics, display_style
    # return dcc.Dropdown( metrics,  id='select_metrics',persistence=True, persistence_type='session')


def create_options_children(options):
    res = []
    for item in options:
        res.append(dbc.ListGroupItem(item))
    return res


def get_group_from_ctx(ctx):
    search_string = "\"group\":\""
    idx = ctx.index(search_string) + len(search_string)
    idx_2 = ctx.index("\"", idx)
    return ctx[idx:idx_2]


# Unfortunately, we can't just use MATCH, as that also requires matching on output
@app.callback(
    Output('legend_data', 'data'),
    Output({'type': 'group-checkbox', 'group': ALL}, 'value'),
    Output({'type': 'child-checkbox', 'group': ALL}, 'value'),
    Output('metric_search', 'value'),
    Input({'type': 'group-checkbox', 'group': ALL}, 'value'),
    Input({'type': 'child-checkbox', 'group': ALL}, 'value'),
    Input('reset_graph', "n_clicks"),
    Input('metric_search', 'value'),
    State({'type': 'group-checkbox', 'group': ALL}, 'options'),
    State({'type': 'child-checkbox', 'group': ALL}, 'options'),
    State({'type': 'group-checkbox', 'group': ALL}, 'value'),
    State({'type': 'child-checkbox', 'group': ALL}, 'value'),
    State('legend_data', 'data'),
    prevent_initial_call=True
)
def group_click(p_selected, c_selected, reset_button, metric_search, p_options, c_options, p_val, c_val, options):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    # print("c_options: ", c_options)
    # print("p_val: ", p_val)
    # print("c_val: ", c_val)
    if ctx == 'reset_graph.n_clicks':
        to_p_val = [[] for _ in range(len(p_val))]
        to_c_val = [[] for _ in range(len(p_val))]
        return [], to_p_val, to_c_val, None

    if ctx == 'metric_search.value':
        print("searched")
        if metric_search is None:
            return options, p_val, c_val, metric_search
        else:
            vals = metric_search.split(" | ")
            metric_name = vals[1] + ',' + vals[0]
            metric = vals[0]
            group = vals[1]
            parent_index = p_options.index([{'label': ' ' + group, 'value': group}])
            if metric_name not in options:
                options.append(metric_name)
                if p_val[parent_index] == []:
                    p_val[parent_index] = group
                c_val[parent_index].append(metric)
            return options, p_val, c_val, metric_search

    group = get_group_from_ctx(ctx)
    parent_index = p_options.index([{'label': ' ' + group, 'value': group}])
    if "\"type\":\"group-checkbox" in ctx:
        print("Group selected")
        child_selection = [option["value"] for option in c_options[parent_index] if p_selected[parent_index]]
        print("options before: ", options)
        options = get_selection_update(group, child_selection, options.copy())
        print("options after: ", options)
        c_val[parent_index] = child_selection
        return options, p_val, c_val, None
    elif "\"type\":\"child-checkbox" in ctx:
        parent_return = []
        if len(c_selected[parent_index]) == len(c_options[parent_index]):
            parent_return = [p_options[parent_index][0]["value"]]
        options = get_selection_update(group, c_val[parent_index], options.copy())
        p_val[parent_index] = parent_return
        return options, p_val, c_val, None
    return options, p_val, c_val, None





@app.callback(
    Output('graph_cnt', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('legend_data', 'data'),
    State('graph_cnt', 'children')
)
def update_graph(n, options, old_container):
    ctx = dash.callback_context
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'interval-component.n_intervals':
        if redisUtil.has_update("metric_graph", reset=True):
            logger.info("new data")
            redisUtil._subscribers["metric_graph"] = False
        else:
            return old_container
    fig = go.Figure()
    if len(options) == 0:
        return []
    for item in options:
        k, v = item.split(',')
        # print(k, v)
        # print('---------------------------')
        add_trace_to(fig, k, v)
    return [dcc.Graph(figure=fig)]
