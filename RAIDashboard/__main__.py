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
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State


from .server import app, dbUtils
from .home_page import get_home_page
from .model_info_page import get_model_info_page
from .certificate_page import get_certificate_page
from .metric_page import get_metric_page
from .metric_info_page import get_metric_info_page
from .single_metric_info_page import get_single_model_info_page
from .certificate_info_page import get_certificate_info_page
from .metric_page_details import get_metric_page_details
from .metric_page_graph import get_metric_page_graph
from .single_metric_view_page import get_single_metric_display
from .setting_page import get_setting_page
from .model_view_page import get_model_view_page
from .data_summary_page import get_data_summary_page
from .model_interpretation_page import get_model_interpretation_page
from .analysis_page import get_analysis_page
from .utils import iconify
import urllib
import sys
from dash.exceptions import PreventUpdate
import signal


def handler(signum, frame):
    print("before closing")
    dbUtils.close()
    print("after closing")
    exit()


signal.signal(signal.SIGINT, handler)

np.seterr(invalid='raise')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "auto",
}


# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


def get_project_list():
    projs = dbUtils.get_projects_list()
    logger.info("projects list acquired : %s" % projs)


def get_sidebar():
    dbUtils._update_projects()
    sidebar = html.Div(
        [
            dbc.Nav(
                [
                    html.Img(src="assets/img/rai_logo_blue3_v2.png", style={
                        "margin-left": "40px", "width": "100px", "height": "60px"
                    }),
                    html.Div([
                        html.Hr(className="nav_div"),
                        html.P("Select the active project"),
                        html.Div(id="dummy_div", style={"display": "no"}),
                        dcc.Dropdown(
                            id="project_selector",
                            options=dbUtils.get_sorted_projects_list(),
                            value=dbUtils._current_project_name,
                            persistence=True,
                            persistence_type="session"), ]),
                    html.Hr(className="nav_div"),
                    html.P("Select the dataset"),
                    html.Div(id="dummy_div_2", style={"display": "no"}),
                    dcc.Dropdown(
                        id="dataset_selector",
                        options=dbUtils.get_dataset_list(),
                        value=dbUtils.get_dataset_list()[0] if len(dbUtils.get_dataset_list()) > 0 else None,
                        persistence=True),
                    html.Hr(className="nav_div"),
                    dbc.NavLink(
                        iconify("Home", "fa-solid fa-home custom-icon", "13px"),
                        href="/", active="exact"),
                    dbc.NavLink(
                        iconify("Settings", "fa-solid fa-gear custom-icon", "15px"),
                        href="/settings", active="exact"),
                    dbc.NavLink(
                        iconify("Project Info", "fa-solid fa-circle-info custom-icon", "15px"),
                        href="/modelInfo", active="exact"),
                    dbc.NavLink(
                        iconify("Metrics Info", "fa-solid fa-file-lines custom-icon", "17px", "2px"),
                        href="/metricsInfo", active="exact"),
                    dbc.NavLink(
                        iconify("Certificates Info", "fa-solid fa-check-double custom-icon", "16px", "1px"),
                        href="/certificateInfo", active="exact"),
                    html.Hr(className="nav_div"),
                    dbc.NavLink(
                        iconify("Metrics Details", "fas fa-table fas-10x custom-icon", "15px"),
                        href="/metrics_details", active="exact"),
                    dbc.NavLink(
                        iconify("Metrics Graphs", "fa-solid fa-chart-gantt custom-icon", "15px"),
                        href="/metrics_graphs", active="exact"),
                    dbc.NavLink(
                        iconify("Individual Metric View", "fa-solid fa-chart-gantt custom-icon", "15px"),
                        href="/individual_metric_view", active="exact"),
                    dbc.NavLink(
                        iconify("Certificates", "fa-solid fa-list-check custom-icon", "15px"),
                        href="/certificates", active="exact"),
                    html.Hr(className="nav_div"),

                    dbc.NavLink(
                        iconify("Model View", "fa-solid fa-eye custom-icon", "13px"),
                        href="/modelView", active="exact"),
                    dbc.NavLink(
                        iconify("Data Summary", "fa-solid fa-newspaper custom-icon", "15px"),
                        href="/dataSummary", active="exact"),
                    dbc.NavLink(
                        iconify("Model Interpretation", "fa-solid fa-microscope custom-icon", "15px"),
                        href="/modelInterpretation", active="exact"),
                    dbc.NavLink(
                        iconify("Analysis", "fa-solid fa-flask-vial custom-icon", "12px"),
                        href="/analysis", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )
    return sidebar


content = html.Div(id="page-content", style=CONTENT_STYLE)


@app.callback(
    Output("dataset_selector", "options"),
    Output("dataset_selector", "value"),
    Output("project_selector", "options"),
    Output("project_selector", "value"),
    Output("main_refresh_reminder", "n_clicks"),
    Input('project_selector', 'value'),
    Input("dataset_selector", "value"),
    Input('main-page-interval-component', 'n_intervals'),
    State("dataset_selector", "options"),
    State("project_selector", "options"),
    State("project_selector", "value"),
    State("main_refresh_reminder", "n_clicks")
)
def render_page_content(value, dataset_value, interval, dataset_options, project_options, project_val, r_reminder):
    r_reminder = 0
    ctx = dash.callback_context
    reload_required = False

    if project_options == [] and len(dbUtils.get_projects_list()) > 0:
        reload_required = True

    dbUtils._update_projects()
    project_options = dbUtils.get_projects_list()

    if dataset_options == [] and len(dbUtils.get_dataset_list()) > 0:
        dataset_options = dbUtils.get_dataset_list()
        dataset_value = dataset_options[0]
        dbUtils.set_current_dataset(dataset_value)
        reload_required = True

    if any("project_selector.value" in i["prop_id"] for i in ctx.triggered):
        dbUtils.set_current_project(value)
        print("Current project: ", dbUtils._current_project_name)
        dataset_options = dbUtils.get_dataset_list()
        dataset_value = dataset_options[0] if len(dataset_options) > 0 else None
        dbUtils.set_current_dataset(dataset_value)
        reload_required = True

    elif any("dataset_selector.value" in i["prop_id"] for i in ctx.triggered):
        dbUtils.set_current_dataset(dataset_value)
        reload_required = True

    if reload_required:
        r_reminder = 1

    if not any("project_selector.value" in i["prop_id"] for i in ctx.triggered):
        project_val = dbUtils._current_project_name

    return dataset_options, dataset_value, project_options, project_val, r_reminder


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("url", "search"),
    Input("main_refresh_reminder", "n_clicks"),
)
def change_page(pathname, search, reminder):
    ctx = dash.callback_context.triggered
    if len(ctx) == 1 and "main_refresh_reminder" in ctx[0]['prop_id'] and reminder == 0:
        raise PreventUpdate

    if search:
        parsed = urllib.parse.urlparse(search)
        parsed_dict = urllib.parse.parse_qs(parsed.query)
    if pathname == "/":
        return get_home_page()
    elif pathname == "/settings":
        return get_setting_page()
    elif pathname == "/metrics":
        return get_metric_page()
    elif pathname == "/metrics_details":
        return get_metric_page_details()
    elif pathname == "/metrics_graphs":
        return get_metric_page_graph()
    elif pathname == "/individual_metric_view":
        return get_single_metric_display()
    elif pathname == "/certificates":
        return get_certificate_page()
    elif pathname == "/modelInfo":
        return get_model_info_page()
    elif pathname == "/single_metric_info/":
        return get_single_model_info_page(parsed_dict["g"][0], parsed_dict["m"][0])
    elif pathname == "/metricsInfo":
        return get_metric_info_page()
    elif pathname == "/certificateInfo":
        return get_certificate_info_page()
    elif pathname == "/modelView":
        return get_model_view_page()
    elif pathname == "/dataSummary":
        return get_data_summary_page()
    elif pathname == "/modelInterpretation":
        return get_model_interpretation_page()
    elif pathname == "/analysis":
        return get_analysis_page()
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    project_list = dbUtils.get_sorted_projects_list()
    print(project_list)
    dbUtils.set_current_project(project_list[0]) if len(project_list) > 0 else None
    if len(dbUtils.get_dataset_list()) > 0:
        dbUtils.set_current_dataset(dbUtils.get_dataset_list()[0])
    app.layout = html.Div([
        dcc.Location(id="url"),
        dcc.Interval(id='main-page-interval-component', interval=2500),
        dbc.Button(id='main_refresh_reminder', style={"display": "None"}),
        get_sidebar(),
        content])
    app.run_server(debug=True)
    dbUtils.close()
