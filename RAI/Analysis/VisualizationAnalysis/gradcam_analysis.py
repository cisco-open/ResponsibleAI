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
import random
import cv2
from RAI.dataset import IteratorData, NumpyData
from RAI.AISystem import AISystem
from RAI.Analysis import Analysis
import os
from dash import html, dcc
import torch
import torch.nn as nn
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


class GradCamAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        all_classes = None
        use_subset = True

        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.data = self.ai_system.get_data(self.dataset)
        self.tag = tag

        self.model = ai_system.model
        self.net = self.model.agent
        self.net.eval()
        self.gradcamModel = GradCAMModel(self.net)
        self.all_classes = self.model.output_features[0].values if all_classes is None else all_classes
        self.n_classes = len(self.all_classes)
        self.all_classes = [self.all_classes[i] for i in range(self.n_classes)]
        self.output_map = self.model.output_features[0].values

        if use_subset:
            classes_subset = [self.all_classes[i] for i in range(self.n_classes)]
            max_classes = 3
            random.shuffle(classes_subset)
            classes_subset = classes_subset[:max_classes]
            self.all_classes = classes_subset

        self.all_classes_num = [i for i in self.output_map if self.output_map[i] in self.all_classes]
        self.n_classes = len(self.all_classes)
        self.n_img = 2
        self.max_progress_tick = self.n_classes * self.n_img * 2 * 3 + 1

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        self.progress_tick()
        self.setGradcamImgs()
        self.setHeatmap()
        self.visualize()

    def setGradcamImgs(self):
        data = self.ai_system.get_data(self.dataset)
        if isinstance(data, NumpyData):
            self.set_gradcam_imgs_numpy()
        elif isinstance(data, IteratorData):
            self.set_gradcam_imgs_iterator()

    def set_gradcam_imgs_iterator(self):
        counts = {i: [0, 0] for i in self.all_classes}
        # counts[i, 0] counts correct samples, counts[i,1] counts wrong samples of class i
        self.gradcamImgs = {c: {"correct": [], "wrong": [], "correct_img": [], "wrong_img": [], "idx": c_idx} for
                            c_idx, c in enumerate(self.all_classes)}
        data = self.ai_system.get_data(self.dataset)
        data.reset()
        while data.next_batch():
            imgs = data.rawX
            if all(sum(counts[i]) == self.n_img * 2 for i in counts):
                break
            imgs = imgs.squeeze()

            # if there are no relevant images, or all images have been sampled, move on
            if not any(i in self.all_classes_num and sum(counts[self.output_map[i]]) < self.n_img * 2 for i in data.y):
                continue

            _, predicted = torch.max(self.net(imgs), 1)
            correct = (predicted == data.rawY).numpy()
            for i in range(len(correct)):
                if data.y[i] in self.all_classes_num:
                    c_idx = data.y[i]
                    if correct[i] and counts[self.output_map[data.y[i]]][0] < self.n_img:
                        counts[self.output_map[data.y[i]]][0] += 1
                        self.gradcamImgs[self.output_map[c_idx]]["correct"].append(imgs[i])
                        self.gradcamImgs[self.output_map[c_idx]]["correct_img"].append(data.X[i][0])
                        self.progress_tick()
                    elif not correct[i] and counts[self.output_map[data.y[i]]][1] < self.n_img:
                        counts[self.output_map[data.y[i]]][1] += 1
                        self.gradcamImgs[self.output_map[c_idx]]["wrong"].append(imgs[i])
                        self.gradcamImgs[self.output_map[c_idx]]["wrong_img"].append(data.X[i][0])
                        self.progress_tick()
        print("Grad-CAM Image Setting done")

    def set_gradcam_imgs_numpy(self):
        counts = {i: [0, 0] for i in self.all_classes}
        # counts[i, 0] counts correct samples, counts[i,1] counts wrong samples of class i
        self.gradcamImgs = {c: {"correct": [], "wrong": [], "correct_img": [], "wrong_img": [], "idx": c_idx} for
                            c_idx, c in enumerate(self.all_classes)}
        data = self.ai_system.get_data(self.dataset)
        r = list(range(len(data.y)))
        random.shuffle(r)
        output_fun = self.ai_system.model.predict_fun
        for i in r:
            if not data.y[i] in self.all_classes_num or sum(counts[self.output_map[data.y[i]]]) >= self.n_img * 2:
                continue
            img = data.rawX[i]
            pred = output_fun(img.unsqueeze(0))[0]
            correct = pred == data.y[i]
            c_idx = data.y[i]
            if correct and counts[self.output_map[data.y[i]]][0] < self.n_img:
                counts[self.output_map[data.y[i]]][0] += 1
                self.gradcamImgs[self.output_map[c_idx]]["correct"].append(img)
                self.gradcamImgs[self.output_map[c_idx]]["correct_img"].append(data.X[i][0])
                self.progress_tick()
            elif not correct and counts[self.output_map[data.y[i]]][1] < self.n_img:
                counts[self.output_map[data.y[i]]][1] += 1
                self.gradcamImgs[self.output_map[c_idx]]["wrong"].append(img)
                self.gradcamImgs[self.output_map[c_idx]]["wrong_img"].append(data.X[i][0])
                self.progress_tick()
        print("Grad-CAM Image Setting done")

    def setHeatmap(self):
        """
        Code based on https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
        """
        self.gradcamResults = {
            c: {
                "correct_heatmap": [],
                "correct_img": self.gradcamImgs[c]["correct_img"],
                "wrong_heatmap": [],
                "wrong_img": self.gradcamImgs[c]["wrong_img"],
            } for c in self.gradcamImgs
        }

        for c in self.gradcamImgs:
            c_dict = self.gradcamImgs[c]
            c_idx = c_dict["idx"]

            for img in c_dict["correct"]:
                img = img.unsqueeze(0)
                pred = self.gradcamModel(img)
                pred[0, c_idx].backward()
                gradients = self.gradcamModel.get_activations_gradient()
                pooled_gradients = torch.mean(gradients, dim=[2, 3])
                activations = self.gradcamModel.get_activations(img).detach()
                activations = activations * pooled_gradients.unsqueeze(-1).unsqueeze(-1)
                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
                heatmap /= torch.max(heatmap)
                self.gradcamResults[c]["correct_heatmap"].append(heatmap.numpy())
                self.progress_tick()

            for img in c_dict["wrong"]:
                img = img.unsqueeze(0)
                pred = self.gradcamModel(img)
                pred[0, c_idx].backward()
                gradients = self.gradcamModel.get_activations_gradient()
                pooled_gradients = torch.mean(gradients, dim=[2, 3])
                activations = self.gradcamModel.get_activations(img).detach()
                activations = activations * pooled_gradients.unsqueeze(-1).unsqueeze(-1)
                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
                heatmap /= torch.max(heatmap)
                self.gradcamResults[c]["wrong_heatmap"].append(heatmap.numpy())
                self.progress_tick()
        print("Grad-CAM Compute Done")

    def img_to_uint8(self, img):
        imin = img.min()
        imax = img.max()
        a = 255 / (imax - imin)
        b = 255 - a * imax
        return (a * img + b).astype(np.uint8)

    def visualize(self):
        self.viz_results = {c: {"correct": [], "wrong": []} for c in self.gradcamImgs}
        for c in self.viz_results:
            c_dict = self.gradcamResults[c]
            for heatmap, img in zip(c_dict["correct_heatmap"], c_dict["correct_img"]):
                img_size = (img.shape[-2], img.shape[-1])
                resized_heatmap = cv2.resize(heatmap, img_size)
                resized_heatmap = self.img_to_uint8(resized_heatmap)
                resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                img = self.img_to_uint8(img)
                superimposed_img = (0.3 * resized_heatmap + 0.7 * img).astype(np.uint8)
                self.viz_results[c]["correct"].append((img, superimposed_img))
                self.progress_tick()

        for c in self.viz_results:
            c_dict = self.gradcamResults[c]
            for heatmap, img in zip(c_dict["wrong_heatmap"], c_dict["wrong_img"]):
                img_size = (img.shape[2], img.shape[1])
                resized_heatmap = cv2.resize(heatmap, img_size)
                resized_heatmap = self.img_to_uint8(resized_heatmap)
                resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                img = self.img_to_uint8(img)
                superimposed_img = (0.3 * resized_heatmap + 0.7 * img).astype(np.uint8)
                self.viz_results[c]["wrong"].append((img, superimposed_img))
                self.progress_tick()
        print("Grad-CAM Visualize Done")

    def to_string(self):
        return "To view results, please open this analysis in the dashboard."

    def to_html(self):
        result = []
        title_div = html.Div(html.H2("Grad-CAM"))
        result.append(title_div)

        tab_content = {}

        for c in self.viz_results:
            img_rows = []
            title_row = html.Tr([html.Td("Correct predicted"), html.Td("Wrongly predicted")])
            for i in range(len(self.viz_results[c]["correct"])):
                correct_data = self.viz_results[c]["correct"][i]
                correct_img, correct_heatmap = np.array(correct_data[0]), np.array(correct_data[1])

                fig_1 = go.Figure(go.Image(z=correct_img))
                fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                fig_1.update_layout(width=200, height=200, margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0))
                fig_graph_1 = html.Div(dcc.Graph(figure=fig_1), style={"display": "inline-block", "padding": "0"})

                fig_2 = go.Figure(go.Image(z=correct_heatmap))
                fig_2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                fig_2.update_layout(width=200, height=200, margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0))
                fig_graph_2 = html.Div(dcc.Graph(figure=fig_2), style={"display": "inline-block", "padding": "0"})
                correct_block = html.Td([fig_graph_1, fig_graph_2])

                wrong_data = self.viz_results[c]["wrong"][i]
                wrong_img, wrong_heatmap = np.array(wrong_data[0]), np.array(wrong_data[1])

                fig_1 = go.Figure(go.Image(z=wrong_img))
                fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                fig_1.update_layout(width=200, height=200, margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0))
                fig_graph_1 = html.Div(dcc.Graph(figure=fig_1), style={"display": "inline-block", "padding": "0"})

                fig_2 = go.Figure(go.Image(z=wrong_heatmap))
                fig_2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                fig_2.update_layout(width=200, height=200, margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0))
                fig_graph_2 = html.Div(dcc.Graph(figure=fig_2), style={"display": "inline-block", "padding": "0"})
                wrong_block = html.Td([fig_graph_1, fig_graph_2])

                img_rows.append(html.Tr([correct_block, wrong_block]))
            table = dbc.Table([title_row] + img_rows)
            tab_content[c] = dbc.Card(dbc.CardBody([table]))
        tabs = dbc.Tabs([dbc.Tab(tab_content[i], label=i) for i in tab_content])
        result.append(tabs)
        res = html.Div(result)
        return res


class GradCAMModel(nn.Module):
    def __init__(self, net):
        super(GradCAMModel, self).__init__()

        self.net = net
        self.hook()

    def hook(self):
        if "torchvision" in self.net.__class__.__module__:
            self.hookTorchvisionModel()
        else:
            self.hookUserDefinedModel()

    def hookTorchvisionModel(self):
        if self.net.__class__.__name__ in ["VGG"]:
            self.features_conv = self.net.features[:36]  # vgg19
        elif self.net.__class__.__name__ in ["RegNet"]:
            self.hookUserDefinedModel()

    def hookUserDefinedModel(self):
        self.features_conv = self.net.features_conv
        self.f1 = self.net.f1
        self.flatten = self.net.flatten
        self.classifier = self.net.classifier

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        x.register_hook(self.activations_hook)

        x = self.f1(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)
