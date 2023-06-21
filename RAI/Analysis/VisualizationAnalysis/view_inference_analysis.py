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


import random
import numpy
import numpy as np
from RAI.AISystem import AISystem
from RAI.Analysis import Analysis
from RAI.dataset import IteratorData, NumpyData
import os
import torch
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
        self.max_progress_tick = self.total_examples

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
        if isinstance(data, NumpyData):
            result = self._get_examples(data.X, data.y, data.rawX)
        elif isinstance(data, IteratorData):
            result = self._get_examples_iterative(data)
        return result

    def _get_examples(self, data_x, data_y, raw_x):
        result = {'X': [] if data_x is not None else None,
                  'y': [] if data_y is not None else None,
                  'output': []}
        size = 0
        output_fun = self._get_output_fun()
        if data_x is not None:
            size = len(data_x)
        elif data_y is not None:
            size = len(data_y)
        r = list(range(size))
        random.shuffle(r)
        r = r[:self.total_examples]
        for example in r:
            if data_y is not None:
                result['y'].append(data_y[example])
            if data_x is not None:
                result['X'].append(data_x[example])
                val = raw_x[example]
                if isinstance(val, torch.Tensor) or isinstance(val, numpy.ndarray):
                    shape = [1]
                    shape.extend(list(val.shape))
                    val = val.reshape(tuple(shape))
                output = output_fun(val)[0]
                result['output'].append(output)
            else:
                result['output'].append(output_fun()[0])
            self.progress_tick()
        return result

    def _get_examples_iterative(self, data: IteratorData):
        result = {'X': [] if data.contains_x else None,
                  'y': [] if data.contains_y else None,
                  'output': []}
        size = 0
        output_fun = self._get_output_fun()
        if not data.next_batch():
            data.reset()
            data.next_batch()
        if data.X is not None:
            size = len(data.X)
        elif data.y is not None:
            size = len(data.y)
        r = list(range(size))
        random.shuffle(r)
        r = r[:self.total_examples]
        for example in r:
            if data.y is not None:
                result['y'].append(data.y[example])
            if data.X is not None:
                result['X'].append(data.X[example])
                val = data.rawX[example]

                # TODO: standardize getting output and converting to output, or just run the model on the whole batch
                if hasattr(val, "reshape"):
                    shape = list(val.shape)
                    shape.insert(0, 1)
                    val = val.reshape(tuple(shape))
                output = output_fun(val)
                result['output'].append(output)
            else:
                result['output'].append(output_fun()[0])
        self.progress_tick()
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
        imin = res.min()
        imax = res.max()
        img = np.transpose(res, (1, 2, 0))
        a = 255 / (imax - imin)
        b = 255 - a * imax
        img = (a * img + b).astype(np.uint8)
        fig = go.Figure(go.Image(z=img))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(width=200, height=200, margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0))
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
                if feature.dtype == "numeric":
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
        self._display_features(header, table_body, [self.output_feature], self.result['output'], "Output")
        for i in range(len(header)):
            header[i] = html.Thead(html.Tr(header[i]))
        for i in range(len(table_body)):
            table_body[i] = html.Tr(table_body[i])

        table = dbc.Table(header + [html.Tbody(table_body)], striped=True, bordered=True)
        result.append(html.Div(table, style={"width": "100%", "height": "100%", "overflow": "scroll"}))
        result = html.Div(result)
        return result
