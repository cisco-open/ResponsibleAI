import logging
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
from dash import dash_table
import dash
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
    print("Get model interpretation page")
    return html.Div([
        html.H4(children='Model Interpretation'),
        get_interpretation()
    ])


def get_interpretation():
    print("Get interpretation")
    global interpretation
    interpretation = redisUtil.get_model_interpretation()
    methods = interpretation.keys()
    div_list = []

    for method in list(methods):
        if method == "gradcam":
            div_list.append(display_gradcam(interpretation[method]))

    return html.Div(div_list)


def display_gradcam(gradcam):
    print("Display gradcam")
    title_div = html.Div("Grad-CAM")
    global c_names
    c_names = list(gradcam.keys())

    # 1. create buttons to select class names
    radios_div = dcc.RadioItems(c_names, c_names[0], id="gradcam-class-radio")

    # 2. display images below when each button is clicked
    gradcam_display_div = html.Div(id="gradcam-display", children=display_gradcam_imgs(c_names[0]))

    return html.Div([title_div, radios_div, gradcam_display_div])




@app.callback(
    Output(component_id="gradcam-display", component_property="children"),
    Input(component_id="gradcam-class-radio", component_property="value"),
    prevent_initial_call=True
)
def display_gradcam_imgs(c_name):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    gradcam = interpretation["gradcam"][c_name]
    
    title_row = html.Tr([html.Td("Correct predicted"), html.Td("Wrongly predicted")], id="gradcam-display-title-row", className="gradcam-display-table-row")
    img_rows = []

    for i in range(5):
        if len(gradcam["correct"]) > i:
            correct_data = gradcam["correct"][i]
            correct_img, correct_heatmap = np.array(correct_data[0]), np.array(correct_data[1])

            fig_1 = go.Figure(go.Image(z=correct_img))
            fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_1.update_layout(width=200, height=200, margin=go.layout.Margin(l=0,r=0,b=0,t=0,pad=0))
            fig_graph_1 = html.Div(dcc.Graph(figure=fig_1), style={"display": "inline-block", "padding": "0"})

            fig_2 = go.Figure(go.Image(z=correct_heatmap))
            fig_2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_2.update_layout(width=200, height=200, margin=go.layout.Margin(l=0,r=0,b=0,t=0,pad=0))
            fig_graph_2 = html.Div(dcc.Graph(figure=fig_2), style={"display": "inline-block", "padding": "0"})

            # correct_img_map = html.Div()
            # print("Correct heatmap ", i, ", ", correct_heatmap.shape)
            correct_block=html.Td([fig_graph_1, fig_graph_2])
        else:
            correct_block=html.Td()
        
        if len(gradcam["wrong"]) > i:
            wrong_data = gradcam["wrong"][i]
            wrong_img, wrong_heatmap = np.array(wrong_data[0]), np.array(wrong_data[1])

            fig_1 = go.Figure(go.Image(z=wrong_img))
            fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_1.update_layout(width=200, height=200, margin=go.layout.Margin(l=0,r=0,b=0,t=0,pad=0))
            fig_graph_1 = html.Div(dcc.Graph(figure=fig_1), style={"display": "inline-block", "padding": "0"})

            fig_2 = go.Figure(go.Image(z=wrong_heatmap))
            fig_2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_2.update_layout(width=200, height=200, margin=go.layout.Margin(l=0,r=0,b=0,t=0,pad=0))
            fig_graph_2 = html.Div(dcc.Graph(figure=fig_2), style={"display": "inline-block", "padding": "0"})

            # wrong_img_map = html.Div([fig_graph_1, fig_graph_2])
            wrong_block=html.Td([fig_graph_1, fig_graph_2])
        else:
            wrong_block=html.Td()

        img_rows.append(html.Tr([correct_block, wrong_block], className="gradcam-display-table-row"))

    table = dbc.Table(
        [title_row] + img_rows
    )
    print("Completed Display Gradcam Images")
    return table







def display_gradcam_imgs_(c_name):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    gradcam = interpretation["gradcam"][c_name]
    
    title_row = html.Tr([html.Td("Correct predicted"), html.Td("Wrongly predicted")])

    img_rows = []

    for i in range(5):
        if len(gradcam["correct"]) > i:
            correct_data = gradcam["correct"][i]
            correct_img, correct_heatmap = np.array(correct_data[0]), np.array(correct_data[1])
            fig_1 = go.Figure(go.Image(z=correct_img))
            fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_graph_1 = dcc.Graph(figure=fig_1)

            fig_2 = go.Figure(go.Image(z=correct_heatmap))
            fig_2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_graph_2 = dcc.Graph(figure=fig_2)

            val = dbc.Table(children=[
                html.Thead(html.Tr([html.Td([html.Th("Image")]), html.Td("Heatmap")])),
                html.Tbody([
                    html.Tr([html.Div(fig_graph_1, style={"display": "inline-block"}), html.Div(fig_graph_2, style={"display": "inline-block"})]),
                    html.Tr([html.Td("text 1"), html.Td("text 2")])])
            ])
            print("Correct heatmap ", i, ", ", correct_heatmap.shape)
            return val
        else:
            correct_img, correct_heatmap = None, None
        
        if len(gradcam["wrong"]) > i:
            wrong_data = gradcam["wrong"][i]
            wrong_img, wrong_heatmap = np.array(wrong_data[0]), np.array(wrong_data[1])
            print("Wrong img ", i, ", ", wrong_img.shape)
            print("wrong heatmap ", i, ", ", wrong_heatmap.shape)
        else:
            wrong_img, wrong_heatmap = None, None

        correct_block = html.Td(html.Div([px.imshow(correct_img), px.imshow(correct_heatmap)]))
        wrong_block = html.Td(html.Div([px.imshow(wrong_img), px.imshow(wrong_heatmap)]))

        img_rows.append(html.Tr([correct_block, wrong_block]))

    table = dbc.Table([
        [title_row] + img_rows
    ])
    print("Completed Display Gradcam Images")
    return table
