import random
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod
from RAI.AISystem import AISystem
from RAI.Analysis import Analysis
from art.estimators.classification import PyTorchClassifier
import os
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


class ViewInferenceAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.total_examples = 5
        self.eps = 0.1

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        data = self.ai_system.get_data(self.dataset)
        self.output_feature = self.ai_system.model.output_features[0]
        self.input_features = self.ai_system.meta_database.features.copy()
        self.task = self.ai_system.task
        self.model = self.ai_system.model
        result = self._get_examples(data.X, data.y)
        return result

    def _get_examples(self, data_x, data_y):
        result = {'X': [] if data_x is not None else None,
                  'y': [] if data_y is not None else None,
                  'output': []}
        size = 0
        output_fun = self._get_output_fun()
        if data_x is not None:
            size = len(data_x)
        elif data_y is not None:
            size = len(data_y)
        r = list(range(size))[:self.total_examples]
        random.shuffle(r)
        for example in r:
            if data_y is not None:
                result['y'].append(data_y[example])
            if data_x is not None:
                result['X'].append(data_x[example])
                val = data_x[example]
                if not isinstance(val[0], np.ndarray) and not isinstance(val[0], list):
                    val = [val]
                result['output'].append(output_fun(val))
            else:
                result['output'].append(output_fun())
        return result

    def _get_output_fun(self):
        result = None
        if 'classification' in self.task or 'regression' in self.task:
            result = self.model.predict_fun
        elif 'generate' in self.task.lower():
            if "text" in self.output_feature.dtype.lower():
                result = self.model.generate_text_fun
            elif "image" in self.output_feature.dtype.lower():
                result = self.model.generate_image_fun
        return result

    def to_string(self):
        result = "\n==== ViewInference ====\nThis Analysis visualizes model inference " \
                 "by showing input and output.\n"
        result += "Please view this analysis in the Dashboard."
        return result

    def to_display_image(self, image):
        shape = list(image.shape)
        shape = tuple(shape[-3:])
        res = image.reshape(shape)
        img = np.transpose(np.uint8(res * 255), (1, 2, 0))
        layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0), width=100, height=100)
        fig = go.Figure(go.Image(z=img), layout=layout)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig_graph = html.Div(dcc.Graph(figure=fig), style={"display": "inline-block", "padding": "0"})
        return fig_graph

    def _display_features(self, header, table_body, features, data, name):
        sty = {"width": "max-content"}
        header[0].append(html.Th(html.B(name, style=sty), colSpan=len(features)))
        header[1].extend([html.Th(html.P(i.name, style=sty)) for i in features])
        for exam_num, example in enumerate(data):
            td_list = []
            if not isinstance(example, list) and not isinstance(example, np.ndarray):
                example = [example]
            for i, feature in enumerate(features):  # TODO: find a cleaner way to get dtype
                if feature.dtype.lower().startswith("int") or feature.dtype.lower().startswith("float") or feature.dtype.lower() == "numeric":
                    if feature.categorical:
                        td_list.append(html.Td(html.P(feature.values[example[i]], style=sty)))
                    else:
                        td_list.append(html.Td(html.P(round(example[i], 4), style=sty)))
                elif feature.dtype.lower() == "text":
                    td_list.append(html.Td(html.P(example[i], style=sty)))
                elif feature.dtype.lower() == "image":
                    td_list.append(self.to_display_image(example[i]))
            table_body[exam_num].extend(td_list)

    def to_html(self):
        result = []
        ts = {"text-align": "center", "display": "block"}
        result.append(html.H1("View Inference Analysis", style=ts))
        result.append(html.P("This Analysis displays input, and output to help understand model inference.", style=ts))
        result.append(html.Br())
        div_count = 1
        if self.result['X'] is not None:
            div_count += 1
        if self.result['y'] is not None:
            div_count += 1
        header = [[], []]
        table_body = [[] for _ in range(self.total_examples)]
        if self.result['X'] is not None:
            self._display_features(header, table_body, self.input_features, self.result['X'], "Data X")
        if self.result['y'] is not None:
            self._display_features(header, table_body, [self.output_feature], self.result['y'], "Data y")
        self._display_features(header, table_body, [self.output_feature], self.result['y'], "Output")
        for i in range(len(header)):
            header[i] = html.Thead(html.Tr(header[i]))
        for i in range(len(table_body)):
            table_body[i] = html.Tr(table_body[i])

        table = dbc.Table(header + [html.Tbody(table_body)], striped=True, bordered=True)
        result.append(html.Div(table, style={"width": "100%", "height": "100%", "overflow": "scroll"}))
        return html.Div(result)
