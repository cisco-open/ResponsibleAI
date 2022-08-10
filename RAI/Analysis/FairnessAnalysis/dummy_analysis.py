from RAI.Analysis import Analysis
from RAI.AISystem import AISystem
import os
from dash import html, dcc
import numpy as np
import plotly.graph_objs as go
import torch
import torchvision
import torchvision.transforms as transforms


class DummyAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None  # calculate result
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag

    def initialize(self):
        pass

    def _compute(self):
        pass

    def to_string(self):
        return "TEST"

    def to_html(self):
        '''       
        res = res.reshape(3, 32, 32)
        img = np.array(res.detach())
        img = np.transpose(np.uint8(img * 255), (1, 2, 0))
        print("img: ", img)
        fig_1 = go.Figure(go.Image(z=img))
        fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig_graph_1 = html.Div(dcc.Graph(figure=fig_1), style={"display": "inline-block", "padding": "0"})
        '''
        result = html.Div([
            html.H4("This Dash HTML is generated at runtime!!"),
            html.Button("Here is a button")], style={})

        return result
