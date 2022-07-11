import logging
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, State
from server import app, redisUtil
from dash import dcc, ALL
from display_types import NumericalElement, FeatureArrayElement, BooleanElement, MatrixElement, DictElement
import metric_view_functions as mvf
logger = logging.getLogger(__name__)

requirements = ["numeric", "feature-array", "boolean", "matrix", "dict"]
prefix = "indiv_"


# TODO: Generalize this with the add traces, combine registry
def populate_display_obj(group, metric):
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


def get_grouped_radio_buttons():
    groups = mvf.get_nonempty_groups(requirements)
    return html.Div([
            html.Details([
                html.Summary([html.P([group], style={"display": "inline-block", "margin-bottom": "0px"})]),
                dcc.RadioItems(
                    id={"type": prefix+"child-checkbox", "group": group},
                    options=[
                        {"label": " " + i, "value": i} for i in mvf.get_valid_metrics(group, requirements)
                    ],
                    value=[],
                    labelStyle={"display": "block"},
                    style={"padding-left": "40px"}
                )]) for group in groups], style={"margin-left": "35%"})


def get_search_and_selection_interface():
    groups = []
    for g in redisUtil.get_metric_info():
        groups.append(g)

    return html.Div(
        dbc.Form([
            dcc.Tabs([
                dcc.Tab(label='Metric Selector', children=[
                    dbc.Row([
                        dbc.Col([get_grouped_radio_buttons()], style={"position": "relative"}),
                        dbc.Col([mvf.get_reset_button(prefix)], style={"position": "relative"}),
                    ], style={"width": "100%", "margin-top": "20px"}),
                ], selected_style=mvf.get_selection_tab_selected_style(), style=mvf.get_selection_tab_style()),
                dcc.Tab(label='Metric Search', children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(mvf.get_search_options(requirements), id=prefix+'metric_search',
                                         value=None, placeholder="Search Metrics"),
                        ], style={"position": "relative"}),
                        dbc.Col([mvf.get_reset_button(prefix)], style={"position": "relative"})
                    ], style={"width": "100%", "margin-top": "20px"}),
                ], selected_style=mvf.get_selection_tab_selected_style(), style=mvf.get_selection_tab_style()),
            ]),
            dbc.Row([dbc.Col([
                html.Br(),
                dbc.Label("Select Tag", html_for="select_metric_tag"),
                dcc.Dropdown([], id=prefix+'select_metric_tag', value=None, placeholder="Select a tag",
                             persistence=True, persistence_type='session')], id=prefix+"select_metric_tag_col",
                style={"display": 'none'})],
                id=prefix+"tag_selector_row")
        ], style=mvf.get_selection_form_style()),
        style=mvf.get_selection_div_style()
    )


def get_single_metric_display():
    return mvf.get_display(prefix, get_search_and_selection_interface())


@app.callback(
    Output(prefix+'legend_data', 'data'),
    Output({'type': prefix+'child-checkbox', 'group': ALL}, 'value'),
    Output(prefix+'metric_search', 'value'),
    Input({'type': prefix+'child-checkbox', 'group': ALL}, 'value'),
    Input(prefix+'reset_graph', "n_clicks"),
    Input(prefix+'metric_search', 'value'),
    State({'type': prefix+'child-checkbox', 'group': ALL}, 'options'),
    State({'type': prefix+'child-checkbox', 'group': ALL}, 'value'),
    State({'type': prefix+'child-checkbox', 'group': ALL}, 'id'),
    State(prefix+'legend_data', 'data'),
    prevent_initial_call=True
)
def update_metric_choice(c_selected, reset_button, metric_search, c_options, c_val, c_ids, options):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    print("CONTEXT: ", ctx)
    ids = [c_ids[i]['group'] for i in range(len(c_ids))]
    if ctx == prefix+'reset_graph.n_clicks':
        to_c_val = [[] for _ in range(len(c_val))]
        return "", to_c_val, None
    if ctx == prefix + 'metric_search.value':
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
    group = mvf.get_group_from_ctx(ctx)
    parent_index = ids.index(group)
    if "\"type\":\"" + prefix +"child-checkbox" in ctx:
        metric = c_val[parent_index]
        options = group + "," + c_val[parent_index]
        c_val = [[] for _ in range(len(c_val))]
        c_val[parent_index] = metric
        print("options: ", options)
        return options, c_val, None
    return options, c_val, None


@app.callback(
    Output(prefix+'graph_cnt', 'children'),
    Output(prefix+'select_metric_tag_col', 'style'),
    Output(prefix+'select_metric_tag', 'options'),
    Output(prefix+'select_metric_tag', 'value'),
    Input(prefix+'interval-component', 'n_intervals'),
    Input(prefix+'legend_data', 'data'),
    Input(prefix+'select_metric_tag', 'value'),
    State(prefix+'graph_cnt', 'children'),
    State(prefix+'select_metric_tag_col', 'style'),
    State(prefix+'select_metric_tag', 'options'),
    State(prefix+'select_metric_tag', 'value')
)
def update_display(n, options, tag_selection, old_graph, old_style, old_children, old_tag_selection):
    ctx = dash.callback_context
    style = {"display": 'none'}
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == prefix + 'interval-component.n_intervals':
        if redisUtil.has_update("metric_graph", reset=True):
            logger.info("new data")
            redisUtil._subscribers["metric_graph"] = False
        else:
            return old_graph, old_style, old_children, old_tag_selection
    elif 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == prefix + 'select_metric_tag.value':
        k, v = options.split(',')
        display_obj, _ = populate_display_obj(k, v)
        tags = display_obj.get_tags()
        return display_obj.display_tag_num(tags.index(tag_selection)), old_style, old_children, tag_selection
    elif options == "" or options == None:
        return [], style, old_children, old_tag_selection
    k, v = options.split(',')
    print(k, v)
    print('---------------------------')
    display_obj, needs_chooser = populate_display_obj(k, v)
    children = []
    selection = None
    if needs_chooser:
        style = {"display": 'block'}
        children = display_obj.get_tags()
        selection = children[-1]
    return display_obj.to_display(), style, children, selection