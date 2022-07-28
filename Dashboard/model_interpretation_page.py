import logging
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output
from server import redisUtil
import dash_daq as daq
import numpy as np
import plotly.express as px
import pandas as pd
from server import app

logger = logging.getLogger(__name__)

c_names = []
interpretation = dict()

def get_model_interpretation_page():
    return html.Div([
        html.H4(children='Model Interpretation'),
        get_interpretation()
    ])


def get_interpretation():
    global interpretation
    interpretation = redisUtil.get_model_interpretation()
    methods = interpretation.keys()
    div_list = []

    for method in list(methods):
        if method == "gradcam":
            div_list.append(display_gradcam(interpretation[method]))

    return html.Div(div_list)


def display_gradcam(gradcam):
    title_div = html.Div("Grad-CAM")
    global c_names
    c_names = list(gradcam.keys())

    # 1. create buttons to select class names
    radios_div = dcc.RadioItems(c_names, c_names[0], id="gradcam-class-radio")

    # 2. display images below when each button is clicked
    gradcam_display_div = html.Div(id="gradcam-display", children=[])

    return html.Div([title_div, radios_div, gradcam_display_div])


@app.callback(
    Output(component_id="gradcam-display", component_property="children"),
    Input(component_id="gradcam-class-radio", component_property="value"),
)
def display_gradcam_imgs(c_name):
    gradcam = interpretation["gradcam"][c_name]
    
    title_row = html.Tr([html.Td("Correct predicted"), html.Td("Wrongly predicted")])

    img_rows = []
    for i in range(5):
        if len(gradcam["correct"]) > i:
            correct_data = gradcam["correct"][i]
            correct_img, correct_heatmap = np.array(correct_data[0]), np.array(correct_data[1])
        else:
            correct_img, correct_heatmap = None, None
        if len(gradcam["wrong"]) > i:
            wrong_data = gradcam["wrong"][i]
            wrong_img, wrong_heatmap = np.array(wrong_data[0]), np.array(wrong_data[1])
        else:
            wrong_img, wrong_heatmap = None, None

        correct_block = html.Td(html.Div([px.imshow(correct_img),px.imshow(correct_heatmap)]))
        wrong_block = html.Td(html.Div([px.imshow(wrong_img),px.imshow(wrong_heatmap)]))

        img_rows.append(html.Tr([correct_block, wrong_block]))

    table = dbc.Table([
        [title_row] + img_rows
    ])
    return table
