import logging
import dash_bootstrap_components as dbc
from dash import html, dcc
from server import redisUtil
import dash_daq as daq
import numpy as np
import plotly.express as px
import pandas as pd

logger = logging.getLogger(__name__)

def get_data_summary_page():
    return html.Div([
        html.H4(children='Data Summary'),
        get_summary()
    ])


def get_label_str(labels, label_name_dict):
    if label_name_dict == None or label_name_dict == "":
        labels_str = ",".join(str(i) for i in labels)
    else:
        label_str = ""
        for i, label in enumerate(label_name_dict):
            label_str += f"{label}({label_name_dict[label]})"
            if i < len(label_name_dict) - 1:
                label_str += ", "
    return label_str


def get_summary():
    data_summary = redisUtil.get_data_summary()

    label_name = data_summary["label_name"]
    target = data_summary["pred_target"]
    labels = data_summary["labels"]

    label_str = get_label_str(labels, label_name)
    
    train_label_dist = data_summary["label_dist"]["train"]
    test_label_dist = data_summary["label_dist"]["test"]
    train_label_df = pd.DataFrame({
        "label": train_label_dist.keys(),
        "freq": train_label_dist.values()
    })
    test_label_df = pd.DataFrame({
        "label": test_label_dist.keys(),
        "freq": test_label_dist.values()
    })

    train_hist = px.bar(train_label_df, x="label", y="freq", labels={})
    test_hist = px.bar(test_label_df, x="label", y="freq", labels={})
    train_hist.update_layout({"margin":{"l":0, "r":0, "t":0, "b":0}})
    train_hist.update_layout(xaxis={"visible": True, "showticklabels": True, "title": None}, yaxis={"visible": False, "showticklabels": False})
    test_hist.update_layout({"margin":{"l":0, "r":0, "t":0, "b":0}})
    test_hist.update_layout(xaxis={"visible": True, "showticklabels": True, "title": None}, yaxis={"visible": False, "showticklabels": False})

    row_target = html.Tr([
        html.Td("Target", className="data-summary-title"),
        html.Td(target, className="data-summary-content"),
    ], className="data-summary-main-row")
    row_label = html.Tr([
        html.Td("Labels", className="data-summary-title"), 
        html.Td(label_str, className="data-summary-content")
    ], className="data-summary-main-row")
    
    row_histogram = html.Tr([
        html.Td("Label Distribution", className="data-summary-title"), 
        html.Td(get_histogram_cell(train_hist, test_hist))
    ], className="data-summary-main-row")

    return dbc.Table(
        [row_target, row_label, row_histogram],
        striped=True,
        borderless=True,
    )

def get_histogram_cell(train_hist, test_hist):
    return dbc.Table([
        html.Tr([
            html.Td("Train Data", className="data-summary-content-hist-title"), 
            html.Td("Test Data", className="data-summary-content-hist-title")
        ]),
        html.Tr([
            html.Td([
                dcc.Graph(figure=train_hist, className="data-summary-content-hist", config={"responsive": True})
            ], className="data-summary-content-half"),
            html.Td([
                dcc.Graph(figure=test_hist, className="data-summary-content-hist", config={"responsive": True})
            ], className="data-summary-content-half")
        ])
    ], borderless=True, style={"margin": 0})