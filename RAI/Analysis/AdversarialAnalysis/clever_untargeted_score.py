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

import numpy as np
import torch
from RAI.AISystem import AISystem
from RAI.Analysis import Analysis
from RAI.dataset import IteratorData, NumpyData
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
import os
from dash import html, dcc, dash_table
import plotly.graph_objs as go
import random


class CleverUntargetedScore(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.EXAMPLES_PER_CLASS = 5
        self.R_L1 = 40
        self.R_L2 = 2
        self.R_LI = 0.1

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        output_features = self.ai_system.model.output_features[0].values
        self.output_features = output_features.copy()
        numClasses = len(output_features)

        self.max_progress_tick = self.EXAMPLES_PER_CLASS * numClasses + 2
        self.progress_tick()

        data = self.ai_system.get_data(self.dataset)
        shape = None
        # This should also be done per class
        result['clever_u_l1'] = {i: [] for i in output_features}
        result['clever_u_l2'] = {i: [] for i in output_features}
        result['clever_u_li'] = {i: [] for i in output_features}

        if isinstance(data, NumpyData):
            balanced_classifications = self._get_balanced_correct_classifications(self.ai_system.model.predict_fun, data.X, data.y, output_features)
            shape = data.image[0].shape
        elif isinstance(data, IteratorData):
            balanced_classifications, shape = self._get_balanced_correct_classifications_iterative(
                self.ai_system.model.predict_fun, data, output_features)
        classifier = PyTorchClassifier(model=self.ai_system.model.agent, loss=self.ai_system.model.loss_function,
                                       optimizer=self.ai_system.model.optimizer, input_shape=shape, nb_classes=numClasses)
        result['total_images'] = 0

        self.progress_tick()

        for target_class in balanced_classifications:
            result['total_images'] += len(balanced_classifications[target_class])
            for example in balanced_classifications[target_class]:
                result['clever_u_l1'][target_class].append(clever_u(classifier, example, 10, 5, self.R_L1, norm=1, pool_factor=3, verbose=False))
                result['clever_u_l2'][target_class].append(clever_u(classifier, example, 10, 5, self.R_L2, norm=2, pool_factor=3, verbose=False))
                result['clever_u_li'][target_class].append(clever_u(classifier, example, 10, 5, self.R_LI, norm=np.inf, pool_factor=3, verbose=False))
                self.progress_tick()
        result['total_classes'] = len(result['clever_u_l1'])
        return result

    def _get_balanced_correct_classifications(self, predict_fun, xData, yData, class_values):
        result_balanced = {i: [] for i in class_values}
        total_images = len(class_values) * self.EXAMPLES_PER_CLASS
        added = 0
        r = list(range(len(yData)))
        random.shuffle(r)
        for i in r:
            if len(result_balanced[yData[i]]) < self.EXAMPLES_PER_CLASS:
                pred = predict_fun(torch.Tensor(xData[i]))[0]
                if pred == yData[i]:
                    result_balanced[yData[i]].append(xData[i][0])
                    added += 1
                    if added >= total_images:
                        break
        return result_balanced

    def _get_balanced_correct_classifications_iterative(self, predict_fun, data: IteratorData, class_values):
        result_balanced = {i: [] for i in class_values}
        total_images = len(class_values) * self.EXAMPLES_PER_CLASS
        added = 0
        shape = None
        data.reset()
        while data.next_batch() and added < total_images:
            if shape is None:
                shape = data.image[0]
            r = list(range(len(data.y)))
            random.shuffle(r)
            for i in r:
                if len(result_balanced[data.y[i]]) < self.EXAMPLES_PER_CLASS:
                    pred = predict_fun(torch.Tensor(data.X[i]))[0]
                    if pred == data.y[i]:
                        result_balanced[data.y[i]].append(data.X[i][0])
                        added += 1
                        if added >= total_images:
                            break
            return result_balanced, shape

    def _get_selection(self, misclassifications):
        offset = int(len(misclassifications) / self.MAX_COMPUTES)
        to_compute = [misclassifications[i] for i in range(0, len(misclassifications), offset)]
        return to_compute

    def _result_stats(self, res):
        return f"Average Value {sum(res) / len(res)} , Minimum Value: {min(res)} , Maximum Value: {max(res)}"

    def to_string(self):
        result = "\n==== CLEVER Untargeted Score Analysis ====\nCLEVER Score is an attack independent robustness metric " \
                 "which can be used to evaluate any neural network.\nCLEVER scores provide a lower bound for adversarial " \
                 "attacks of various norms.\n"
        result += "This evaluation computed the CLEVER score across " + str(self.result['total_images']) + " examples.\n"
        result += "CLEVER untargeted scores describe attacks where the adversary attempts to trick the classifier to pick " \
                  "any incorrect class.\n"
        result += "L1 Perturbations describes the sum of the perturbation size.\n"
        for val in self.result['clever_u_l1']:
            result += "The Untargeted CLEVER L1 score to trick the classifier from correctly picking class " + self.output_features[val] + " is: \n" \
                      + self._result_stats(self.result['clever_u_l1'][val]) + "\n"
        result += "\nL2 Perturbations describes the manhattan distance between the input before and after perturbation.\n"
        for val in self.result['clever_u_l2']:
            result += "The Untargeted CLEVER L2 score to trick the classifier from correctly picking class " + self.output_features[val] + " is: \n" \
                      + self._result_stats(self.result['clever_u_l2'][val]) + "\n"
        result += "\nL-inf Perturbations describes the maximum size of a perturbation.\n"
        for val in self.result['clever_u_li']:
            result += "The Untargeted CLEVER L-inf score to trick the classifier from correctly picking class " + self.output_features[val] + " is: \n" \
                      + self._result_stats(self.result['clever_u_li'][val]) + "\n"
        return result

    def get_avg(self):
        pass

    def i_to_data(self, i, data_dict):
        return {0: self.output_features[i],
                1: round(sum(data_dict[i]) / len(data_dict[i]), 4),
                2: round(min(data_dict[i]), 4),
                3: round(max(data_dict[i]), 4)}

    def rand_initialize(self):
        res = {i: [] for i in self.output_features}
        for i in res:
            res[i] = [random.random() for _ in range(self.EXAMPLES_PER_CLASS)]
        return res

    def get_fancy_figure(self, data_dict):

        avgs = [sum(data_dict[i]) / len(data_dict[i]) if data_dict[i] else 0 for i in data_dict]
        maxs = [max(data_dict[i]) if data_dict[i] else 0 for i in data_dict]
        mins = [min(data_dict[i]) if data_dict[i] else 0 for i in data_dict]
        layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))
        fig = go.Figure([go.Bar(x=[self.output_features[i] for i in self.output_features],
                                y=avgs,
                                error_y={'type': 'data', 'symmetric': False,
                                         'array': [round(maxs[i] - avgs[i], 4) for i in range(len(avgs))],
                                         'arrayminus': [round(avgs[i] - mins[i], 4) for i in range(len(avgs))]})],
                        layout=layout)
        fig_graph = html.Div(dcc.Graph(figure=fig), style={"display": "block", "margin": "0 auto 0 auto", "width": "80%"})
        return fig_graph

    def get_table(self, cols, data_dict):
        return dash_table.DataTable(columns=cols, data=[self.i_to_data(i, data_dict) for i in data_dict], fill_width=False)

    def to_html(self):
        result = []
        l1_score = self.result['clever_u_l1']  # self.rand_initialize()
        l2_score = self.result['clever_u_l2']
        li_score = self.result['clever_u_li']
        total_images = self.result['total_images']
        total_classes = self.result['total_classes']

        ts = {"text-align": "center", "display": "block"}
        result.append(html.H1("CLEVER Untargeted Score Analysis", style=ts))
        result.append(html.P("CLEVER Score is an attack independent robustness metric which can be used to evaluate "
                             "any neural network.", style=ts))
        result.append(html.P("CLEVER scores provide a lower bound for adversarial attacks of various norms.", style=ts))
        result.append(html.P("CLEVER untargeted scores describe attacks where the adversary attempts to trick the "
                             "classifier to pick any incorrect class", style=ts))
        result.append(html.Br())
        result.append(html.B(
            f"For this analysis, {total_images} images were evenly selected across {total_classes} classes.",
            style=ts))
        result.append(html.B("For a subset of images, Untargeted Clever Scores were calculated to see how easily the "
                             "classifier can be tricked from correctly picking that class.", style=ts))
        result.append(html.Br())
        result.append(html.H4("L1 Perturbations", style=ts))
        result.append(html.P("L1 Perturbations describes the sum of the perturbation size.", style=ts))

        fig_graph = self.get_fancy_figure(l1_score)
        result.append(fig_graph)
        result.append(html.Br())
        result.append(html.Br())
        result.append(html.H4("L2 Perturbations", style=ts))
        result.append(html.P("L2 Perturbations describes the manhattan distance between the input before and after perturbation.", style=ts))

        fig_graph = self.get_fancy_figure(l2_score)
        result.append(fig_graph)
        result.append(html.Br())
        result.append(html.Br())
        result.append(html.H4("L-inf Perturbations", style=ts))
        result.append(html.P("L-inf Perturbations describes the maximum size of a perturbation.", style=ts))

        fig_graph = self.get_fancy_figure(li_score)
        result.append(fig_graph)
        return html.Div(result)
