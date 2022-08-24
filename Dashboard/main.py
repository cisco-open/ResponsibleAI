"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""

import logging
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
from home_page import get_home_page
from model_info_page import get_model_info_page
from certificate_page import get_certificate_page
from metric_page import get_metric_page
from metric_info_page import get_metric_info_page
from single_metric_info_page import get_single_model_info_page
from certificate_info_page import get_certificate_info_page
from metric_page_details import get_metric_page_details
from metric_page_graph import get_metric_page_graph
from single_metric_view_page import get_single_metric_display
from setting_page import get_setting_page
from model_view_page import get_model_view_page
from data_summary_page import get_data_summary_page
from model_interpretation_page import get_model_interpretation_page
from analysis_page import get_analysis_page
from utils import iconify
import urllib
import sys
from dash.exceptions import PreventUpdate


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
    projs = redisUtil.get_projects_list()
    logger.info("projects list acquired : %s" % projs)


def get_sidebar():
    sidebar = html.Div(
        [
            html.H2(["RAI", html.Img(src="./assets/img/rai_logo.png", style={"float": "right", "width": "62px", "height": "80px"})], className="display-4"),
            html.P("A framework for Responsible AI development", className="small"),
            html.Hr(),
            dbc.Nav(
                [ 
                    html.P("Select the active project"),
                    html.Div(id="dummy_div", style={"display": "no"}),
                    dcc.Dropdown(
                        id="project_selector",
                        options=redisUtil.get_projects_list(),
                        value=redisUtil._current_project_name,
                        persistence=True,
                        persistence_type="session"),
                    html.Hr(),
                    html.P("Select the dataset"),
                    html.Div(id="dummy_div_2", style={"display": "no"}),
                    dcc.Dropdown(
                        id="dataset_selector",
                        options=redisUtil.get_dataset_list(),
                        value=redisUtil.get_dataset_list()[0] if len(redisUtil.get_dataset_list()) > 0 else None,
                        persistence=True),
                    html.Hr(),
                    dbc.NavLink(  
                        iconify("Home", "fa-solid fa-home", "25px"),
                        href="/", active="exact"),
                    dbc.NavLink(  
                        iconify("Settings", "fa-solid fa-gear", "25px"),
                        href="/settings", active="exact"),
                    html.Hr(),
                    dbc.NavLink( 
                        iconify("Metrics Details", "fas fa-table fas-10x", "18px"),
                        href="/metrics_details", active="exact"),
                    dbc.NavLink(
                        iconify("Metrics Graphs", "fa-solid fa-chart-gantt", "18px"),
                        href="/metrics_graphs", active="exact"),
                    dbc.NavLink(
                        iconify("Individual Metric View", "fa-solid fa-chart-gantt", "18px"),
                        href="/individual_metric_view", active="exact"),
                    dbc.NavLink(
                        iconify("Certificates", "fa-solid fa-list-check", "45px"),
                        href="/certificates", active="exact"),
                    html.Hr(),
                    dbc.NavLink(
                        iconify("Project Info", "fa-solid fa-circle-info", "55px"),
                        href="/modelInfo", active="exact"),
                    dbc.NavLink(
                        iconify("Metrics Info", "fa-solid fa-file-lines", "50px"),
                        href="/metricsInfo", active="exact"),
                    dbc.NavLink(
                        iconify("Model View", "fa-solid fa-eye", "50px"),
                        href="/modelView", active="exact"),
                    dbc.NavLink( 
                        iconify("Certificates Info", "fa-solid fa-check-double", "20px"),
                        href="/certificateInfo", active="exact"),
                    dbc.NavLink( 
                        iconify("Data Summary", "fa-solid fa-newspaper", "20px"),
                        href="/dataSummary", active="exact"),
                    dbc.NavLink( 
                        iconify("Model Interpretation", "fa-solid fa-microscope", "20px"),
                        href="/modelInterpretation", active="exact"),
                    dbc.NavLink(
                        iconify("Analysis", "fa-solid fa-flask-vial", "20px"),
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

    if project_options == [] and len(redisUtil.get_projects_list()) > 0:
        reload_required = True

    project_options = redisUtil.get_projects_list()

    if dataset_options == [] and len(redisUtil.get_dataset_list()) > 0:
        dataset_options = redisUtil.get_dataset_list()
        dataset_value = dataset_options[0]
        redisUtil.set_current_dataset(dataset_value)
        reload_required = True

    if any("project_selector.value" in i["prop_id"] for i in ctx.triggered):
        redisUtil.set_current_project(value)
        print("Current project: ", redisUtil._current_project_name)
        dataset_options = redisUtil.get_dataset_list()
        dataset_value = dataset_options[0] if len(dataset_options) > 0 else None
        redisUtil.set_current_dataset(dataset_value)
        reload_required = True

    elif any("dataset_selector.value" in i["prop_id"] for i in ctx.triggered):
        redisUtil.set_current_dataset(dataset_value)
        reload_required = True

    if reload_required:
        r_reminder = 1

    if not any("project_selector.value" in i["prop_id"] for i in ctx.triggered):
        project_val = redisUtil._current_project_name

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
    redisUtil.initialize(subscribers={"metric_detail", "metric_graph", "certificate", "analysis_update"})
    project_list = redisUtil.get_projects_list()
    redisUtil.set_current_project(project_list[0]) if len(project_list) > 0 else None
    if len(redisUtil.get_dataset_list()) > 0:
        redisUtil.set_current_dataset(redisUtil.get_dataset_list()[0])
    app.layout = html.Div([dcc.Location(id="url"), dcc.Interval(
                id='main-page-interval-component', interval=1 * 2500, n_intervals=0),
                dbc.Button(id='main_refresh_reminder', style={"display": "None"}), get_sidebar(), content])

    app.run_server(debug=False)
    redisUtil.close()
