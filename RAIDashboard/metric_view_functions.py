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
from dash import html
from .server import dbUtils
from dash import dcc
import dash_bootstrap_components as dbc
from .display_types.display_factory import is_compatible

logger = logging.getLogger(__name__)

__all__ = ['get_nonempty_groups', 'get_valid_metrics', 'get_search_options', 'get_graph', 'get_display',
           'get_reset_button', 'get_graph_update_purpose']


def get_nonempty_groups(requirements):
    metric_info = dbUtils.get_metric_info()
    valid_groups = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and is_compatible(metric_info[group][metric]["type"], requirements):
                valid_groups.append(group)
                break
        if group in ["Custom", "Certificates"]:
            valid_groups.append(group)
    return valid_groups


def get_valid_metrics(group, requirements):
    metric_info = dbUtils.get_metric_info()
    valid_metrics = []
    for metric in metric_info[group]:
        if "type" in metric_info[group][metric] and is_compatible(metric_info[group][metric]["type"], requirements):
            valid_metrics.append(metric)
        elif group in ['Custom', 'Certificates'] and metric != 'meta':
            valid_metrics.append(metric)
    return valid_metrics


def get_search_options(requirements):
    metric_info = dbUtils.get_metric_info()
    valid_searches = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and is_compatible(metric_info[group][metric]["type"], requirements):
                label = f"{metric_info[group][metric]['display_name']} | {metric_info[group]['meta']['display_name']}"
                valid_searches.append({
                    'label': label,
                    'value': metric + " | " + group})
    return valid_searches


def get_graph(prefix):
    return html.Div(
        html.Div(id=prefix + 'graph_cnt', children=[], style={"margin": "1"}),
        style={
            "border-width": "thin",
            "border-color": "LightGray",
            "border-style": "solid",
            "border-radius": "3px",
            "margin": "1"})


def get_display(prefix, selectors):
    return html.Div([
        dcc.Store(id=prefix + 'legend_data', storage_type='memory'),
        dcc.Interval(
            id=prefix + 'interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0),
        html.Div(html.Div([selectors])),
        get_graph(prefix),
        html.Div(id=prefix + "metric_info", children=[])])


def get_reset_button(prefix):
    return dbc.Button("Reset Graph", id=prefix + "reset_graph", color="secondary",
                      style={"position": "absolute", "bottom": "0"})


def analysis_update_cause(ctx, prefix):
    is_time_update = any(prefix + 'interval-component.n_intervals' in i['prop_id'] for i in ctx.triggered)
    is_button_click = any('run_analysis_button.n_clicks' in i['prop_id'] for i in ctx.triggered)
    is_value = any('analysis_selector.value' == i['prop_id'] for i in ctx.triggered)
    return is_time_update, is_button_click, is_value


def get_graph_update_purpose(ctx, prefix):
    is_new_data = any(prefix + 'legend_data.data' in i['prop_id'] for i in ctx.triggered)
    is_time_update = any(prefix + 'interval-component.n_intervals' in i['prop_id'] for i in ctx.triggered)
    is_new_tag = any(prefix + 'select_metric_tag.value' in i['prop_id'] for i in ctx.triggered)
    return is_new_data, is_time_update, is_new_tag


# ============ STYLE RELATED ============

def get_selection_form_style():
    return {"background-color": "rgb(240,250,255)", "width": "100%", "border": "solid",
            "border-color": "silver", "border-radius": "5px", "padding": "10px 50px 10px 50px"}


def get_selection_div_style():
    return {"margin": "2px"}


def get_selection_tab_style():
    return {
        'borderBottom': '1px solid #d6d6d6',
        'padding': '6px',
        'fontWeight': 'bold'}


def get_selection_tab_selected_style():
    return {
        'borderTop': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': '#119DFF',
        'color': 'white',
        'padding': '6px'}
