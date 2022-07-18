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


def get_summary():
    data_summary = redisUtil.get_data_summary()

    labels = data_summary["labels"]
    labels_str = ",".join(str(i) for i in labels)
    
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

    train_hist = px.bar(train_label_df, x="label", y="freq")
    test_hist = px.bar(test_label_df, x="label", y="freq")

    print(data_summary)
    labelDiv = html.Div([
        html.Div("Labels"),
        html.Div(labels_str)
    ])
    histDiv = html.Div([
        html.Div("Distribution"),
        dcc.Graph(figure=train_hist),
        dcc.Graph(figure=test_hist),
    ])


    return html.Div([labelDiv, histDiv])

