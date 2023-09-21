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
from dash import Input, Output, html, State
from .server import app, dbUtils
from dash import dcc, ALL
import plotly.graph_objs as go
from .display_types import get_display
import RAIDashboard.metric_view_functions as mvf
from dash.exceptions import PreventUpdate
logger = logging.getLogger(__name__)

requirements = ["Traceable"]
prefix = "grouped_"
selector_height = "320px"


def add_trace_to_fig(fig, group, metric):
    dataset = dbUtils.get_current_dataset()
    metric_values = dbUtils.get_metric_values()
    metric_type = dbUtils.get_metric_info()
    if group not in metric_type:
        return
    if metric not in metric_type[group]:
        return
    type = metric_type[group][metric].get("type", "numeric")
    display_obj = get_display(metric, type, dbUtils)
    if display_obj is not None:
        for i, data in enumerate(metric_values):
            data = data[dataset]
            display_obj.append(data[group][metric], data["metadata"]["tag"])
        display_obj.add_trace_to(fig)
    return


# Removes all values from a certain group, and re-adds all of those found in selected
def update_metric_selections(group: str, selected: list, options: list):
    options = [item for item in options if not item.startswith(group + ",")]
    for val in selected:
        options.append(group + "," + val)
    return options


def get_grouped_checklist():
    def entire_group_selected(group):
        return len(options.get(group, [])) == len([i for i in mvf.get_valid_metrics(group, requirements)])
    groups = mvf.get_nonempty_groups(requirements)
    metric_info = dbUtils.get_metric_info()
    options = dbUtils.config_db.get('options', {})
    return html.Div([
        html.Details([
            html.Summary([dcc.Checklist(
                id={"type": prefix + "group-checkbox", "group": group},
                options=[{"label": metric_info[group]['meta']['display_name'], "value": group}],
                value=[group if entire_group_selected(group) else None],
                labelStyle={"display": "inline-block"},
                style={"display": "inline-block"},
                inputStyle={"margin-right": "5px"}
            )]),
            dcc.Checklist(
                id={"type": prefix + "child-checkbox", "group": group},
                options=[
                    {"label": metric_info[group][i]['display_name'], "value": i}
                    for i in mvf.get_valid_metrics(group, requirements)],
                value=options.get(group, []),
                labelStyle={"display": "block"},
                inputStyle={"margin-right": "5px"},
                style={"padding-left": "40px"},
                inputClassName="grouped-checklist-input"
            )],
            open=True if options.get(group) else False,
        ) for group in groups], style={"margin-left": "35%", "height": "100%", "overflow-y": "scroll"})


def get_search_and_selection_interface():
    return html.Div(
        dbc.Form([
            dcc.Tabs([
                dcc.Tab(label='Metric Selector', children=[
                    dbc.Row([
                        dbc.Col([
                            get_grouped_checklist(),
                        ], style={"position": "relative", "height": selector_height}),
                        dbc.Col(html.Br()),
                    ], style={"width": "100%", "margin-top": "20px"}),
                ],
                    selected_style=mvf.get_selection_tab_selected_style(),
                    style=mvf.get_selection_tab_style(),
                    className='metric_selector'),
                dcc.Tab(label='Metric Search', children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                mvf.get_search_options(requirements), id=prefix + 'metric_search',
                                value=None, placeholder="Search Metrics"),
                        ], style={"position": "relative"}),
                        dbc.Col([
                            dbc.Button("Reset Graph", id=prefix + "reset_graph", color="secondary"),
                        ], style={"position": "relative"})
                    ], style={"width": "100%", "margin-top": "20px"})
                ], selected_style=mvf.get_selection_tab_selected_style(), style=mvf.get_selection_tab_style()),
            ]),
        ], style=mvf.get_selection_form_style()),
        style=mvf.get_selection_div_style()
    )


def get_full_interface():
    return html.Div(
        [
            get_search_and_selection_interface(),
            html.Div(
                [
                    dbc.Button(
                        "Reset Graph", id=prefix + "reset_graph", color="secondary",
                        style={"position": "relative", "left": "42%"}),
                ],
                className='grouped_reset_graph_div',
                style={'padding-bottom': '3px'}
            ),
        ]
    )


def get_metric_page_graph():
    return mvf.get_display(prefix, get_full_interface())


def update_metrics_config(func):
    def wrapper(*args, **kwargs):
        options, p_val, c_val, metric_search = func(*args, **kwargs)
        data = {}
        for option in options:
            group, metric = option.split(',')
            if group in data:
                data[group].append(metric)
            else:
                data[group] = [metric]
        dbUtils._metrics_config = data
        dbUtils.config_db['options'] = dbUtils._metrics_config
        return options, p_val, c_val, metric_search
    return wrapper


# Unfortunately, we can't just use MATCH, as that also requires matching on output
@app.callback(
    Output(prefix + 'legend_data', 'data'),
    Output({'type': prefix + 'group-checkbox', 'group': ALL}, 'value'),
    Output({'type': prefix + 'child-checkbox', 'group': ALL}, 'value'),
    Output(prefix + 'metric_search', 'value'),
    Input({'type': prefix + 'group-checkbox', 'group': ALL}, 'value'),
    Input({'type': prefix + 'child-checkbox', 'group': ALL}, 'value'),
    Input(prefix + 'reset_graph', "n_clicks"),
    Input(prefix + 'metric_search', 'value'),
    State({'type': prefix + 'group-checkbox', 'group': ALL}, 'options'),
    State({'type': prefix + 'child-checkbox', 'group': ALL}, 'options'),
    State({'type': prefix + 'group-checkbox', 'group': ALL}, 'value'),
    State({'type': prefix + 'child-checkbox', 'group': ALL}, 'value'),
    State(prefix + 'legend_data', 'data'),
    prevent_initial_call=True
)
@update_metrics_config
def update_metric_choices(p_selected, c_selected, reset_button, metric_search, p_options, c_options, p_val, c_val, options):
    c_val = [el if el is not None else [] for el in c_val]
    p_val = [el if el is not None else [] for el in p_val]
    ctx = dash.callback_context.triggered
    metric_info = dbUtils.get_metric_info()
    options_value = dbUtils.config_db.get('options', {})
    options = [] if options is None else options
    for k, v in options_value.items():
        for item in v:
            options.append(f'{k},{item}')
    options = list(set(options))
    if any(prefix + 'reset_graph.n_clicks' in i["prop_id"] for i in ctx):
        to_p_val = [[] for _ in range(len(p_val))]
        to_c_val = [[] for _ in range(len(p_val))]
        return [], to_p_val, to_c_val, None
    if any(prefix + 'metric_search.value' in i["prop_id"] for i in ctx):
        if metric_search is None:
            return options, p_val, c_val, metric_search
        else:
            vals = metric_search.split(" | ")
            metric_name = vals[1] + ',' + vals[0]
            metric = vals[0]
            group = vals[1]
            parent_index = p_options.index([{'label': metric_info[group]['meta']['display_name'], 'value': group}])
            if metric_name not in options:
                options.append(metric_name)
                if len(c_val[parent_index]) == len(c_options[parent_index]) - 1:
                    p_val[parent_index] = [group]
                c_val[parent_index].append(metric)
            return options, p_val, c_val, metric_search
    group = dash.callback_context.triggered_id["group"]
    parent_index = p_options.index([{'label': metric_info[group]['meta']['display_name'], 'value': group}])
    if any("\"type\":\"" + prefix + "group-checkbox" in i["prop_id"] for i in ctx):
        child_selection = [option["value"] for option in c_options[parent_index] if p_selected[parent_index]]
        options = update_metric_selections(group, child_selection, options.copy())
        c_val[parent_index] = child_selection
        return options, p_val, c_val, None
    elif any("\"type\":\"" + prefix + "child-checkbox" in i["prop_id"] for i in ctx):
        parent_return = []
        if len(c_selected[parent_index]) == len(c_options[parent_index]):
            parent_return = [p_options[parent_index][0]["value"]]
        options = update_metric_selections(group, c_val[parent_index], options.copy())
        p_val[parent_index] = parent_return
        return options, p_val, c_val, None
    return options, p_val, c_val, None


@app.callback(
    Output(prefix + 'graph_cnt', 'children'),
    Input(prefix + 'interval-component', 'n_intervals'),
    Input(prefix + 'legend_data', 'data'),
    State(prefix + 'graph_cnt', 'children')
)
def update_graph(n, options, old_container):
    def options_from_db():
        opt = []
        db_options = dbUtils._metrics_config
        for k, v in db_options.items():
            for item in v:
                opt.append(f'{k},{item}')
        return opt
    ctx = dash.callback_context
    is_new_data, is_time_update, _ = mvf.get_graph_update_purpose(ctx, prefix)
    if is_time_update:
        if dbUtils.has_update("metric_graph", reset=True):
            logger.info("new data")
            dbUtils._subscribers["metric_graph"] = False
            is_new_data = True

    if is_new_data or options_from_db():
        fig = go.Figure()
        if options is None:
            options = options_from_db()
        if len(options) == 0:
            return []
        for item in options:
            k, v = item.split(',')
            add_trace_to_fig(fig, k, v)
        return [dcc.Graph(figure=fig)]

    raise PreventUpdate
