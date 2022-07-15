import logging
import pandas as pd
from dash import html
from server import redisUtil
from dash import dcc
import plotly.express as px
import dash_bootstrap_components as dbc
from display_types.display_factory import is_compatible

logger = logging.getLogger(__name__)

__all__ = ['get_nonempty_groups', 'get_valid_metrics', 'get_search_options', 'get_graph', 'get_display',
           'get_reset_button', 'get_graph_update_purpose']


def get_nonempty_groups(requirements):
    metric_info = redisUtil.get_metric_info()
    valid_groups = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and is_compatible(metric_info[group][metric]["type"], requirements):
                valid_groups.append(group)
                break
    return valid_groups


def get_valid_metrics(group, requirements):
    metric_info = redisUtil.get_metric_info()
    valid_metrics = []
    for metric in metric_info[group]:
        if "type" in metric_info[group][metric] and is_compatible(metric_info[group][metric]["type"], requirements):
            valid_metrics.append(metric)
    return valid_metrics


def get_search_options(requirements):
    metric_info = redisUtil.get_metric_info()
    valid_searches = []
    for group in metric_info:
        for metric in metric_info[group]:
            if "type" in metric_info[group][metric] and is_compatible(metric_info[group][metric]["type"], requirements):
                valid_searches.append({'label': metric_info[group][metric]['display_name'] +
                                                ' | ' + metric_info[group]['meta']['display_name'],
                                       'value': metric + " | " + group})
    return valid_searches


def get_graph(prefix):
    d = {"x": [], "value": [], "tag": [], "metric": []}
    fig = px.line(pd.DataFrame(d), x="x", y="value", color="metric", markers="True")
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
        dcc.Store(id=prefix + 'legend_data', storage_type='local'),
        dcc.Interval(
            id=prefix + 'interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0),
        html.Div(html.Div([selectors])),
        get_graph(prefix),
        html.Div(id=prefix+"metric_info", children=[])])


def get_group_from_ctx(ctx):
    search_string = "\"group\":\""
    idx = ctx.index(search_string) + len(search_string)
    idx_2 = ctx.index("\"", idx)
    return ctx[idx:idx_2]


def get_reset_button(prefix):
    return dbc.Button("Reset Graph", id=prefix + "reset_graph", color="secondary",
                      style={"position": "absolute", "bottom": "0"})


def get_graph_update_purpose(ctx, prefix):
    is_new_data = False
    is_time_update = False
    is_new_tag = False
    for val in ctx.triggered:
        if val['prop_id'] == prefix + 'legend_data.data':
            is_new_data = True
        if val['prop_id'] == prefix + 'select_metric_tag.value':
            is_new_tag = True
        if val['prop_id'] == prefix + 'interval-component.n_intervals':
            is_time_update = True
    return is_new_data, is_time_update, is_new_tag


# ============ STYLE RELATED ============

def get_selection_form_style():
    return {"background-color": "rgb(240,250,255)", "width": "100%", "border": "solid",
            "border-color": "silver", "border-radius": "5px", "padding": "10px 50px 10px 50px"}


def get_selection_div_style():
    return {"margin": "2px", "margin-bottom": "20px"}


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
