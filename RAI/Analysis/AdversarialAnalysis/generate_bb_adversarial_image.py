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


class GenerateBrendelBethgeAdversarialImage(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.total_images = 10
        self.max_progress_tick = self.total_images + 3
        self.eps = 0.1

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        self.progress_tick()

        data = self.ai_system.get_data(self.dataset)
        xData = data.X
        yData = data.y
        output_features = self.ai_system.model.output_features[0].values
        self.output_features = output_features.copy()
        numClasses = len(output_features)
        shape = data.image[0].shape

        self.progress_tick()

        classifier = PyTorchClassifier(model=self.ai_system.model.agent, loss=self.ai_system.model.loss_function,
                                       optimizer=self.ai_system.model.optimizer, input_shape=shape, nb_classes=numClasses)
        correct_classifications = self._get_correct_classifications(self.ai_system.model.predict_fun, xData, yData)
        balanced_classifications = self._balance_classifications_per_class(correct_classifications, yData, output_features)
        input_selections = self._select_random(balanced_classifications)
        attack = FastGradientMethod(estimator=classifier, eps=self.eps, minimal=True, eps_step=0.005, num_random_init=3)
        result['total_images'] = 0
        result['total_classes'] = 0
        result['adv_output'] = {}

        self.progress_tick()

        for target_class in input_selections:
            result['total_images'] += 1
            result['total_classes'] += 1
            og_image = xData[input_selections[target_class]]
            x_adv = attack.generate(x=og_image)
            adv_output = self.ai_system.model.predict_fun(torch.from_numpy(x_adv))
            adv_output = np.argmax(adv_output.detach().numpy(), axis=1)[0]
            result['adv_output'][target_class] = {"image": og_image,
                                    "adversarial": x_adv,
                                    "final_prediction": adv_output}
            self.progress_tick()
        return result

    def _select_random(self, balanced_classifications):
        result = {}
        while len(balanced_classifications) > 0 and len(result) < self.total_images:
            selection = random.choice(list(balanced_classifications.keys()))
            result[selection] = balanced_classifications[selection]
            balanced_classifications.pop(selection)
        return result

    def _get_correct_classifications(self, predict_fun, xData, yData):
        result = []
        for i, example in enumerate(xData):
            pred = predict_fun(torch.Tensor(example))
            if np.argmax(pred.detach().numpy(), axis=1)[0] == yData[i]:
                result.append(i)
        return result

    def _balance_classifications_per_class(self, classifications, yData, class_values):
        result = {i: None for i in class_values}
        r = list(range(len(yData)))
        random.shuffle(r)
        for classification in r:
            if result[yData[classification]] is None:
                result[yData[classification]] = classification
        return result

    def to_string(self):
        result = "\n==== Generate Brendle Bethge Adversarial Image Analysis ====\nThis Analysis uses the Brendle Bethge Method to " \
                 "generate adversarial images.\nOne image is selected across each class (max 10), an an adversarial " \
                 "example is generated for each.\n"
        result += "For this analysis, " + str(self.result['total_images']) + " images were evenly selected across " + \
                  str(self.result['total_classes']) + " classes.\n"
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

    def get_diff(self, org, adv):
        shape = list(org.shape)
        shape = shape[-3:]
        org = org.reshape(shape)
        org = np.transpose(np.uint8(org * 255), (1, 2, 0))
        org = org.astype('int32')
        adv = adv.reshape(shape)
        adv = np.transpose(np.uint8(adv * 255), (1, 2, 0))
        adv = adv.astype('int32')
        diff = np.abs(adv - org)
        diff = diff.astype("uint8")
        layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0), width=100, height=100)
        fig = go.Figure(go.Image(z=diff), layout=layout)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig_graph = html.Div(dcc.Graph(figure=fig), style={"display": "inline-block", "padding": "0"})
        return fig_graph

    def to_html(self):
        result = []
        adv_res = self.result['adv_output']
        total_images = self.result['total_images']
        total_classes = self.result['total_classes']
        ts = {"text-align": "center", "display": "block"}

        result.append(html.H1("Generate Adversarial Image Analysis", style=ts))
        result.append(html.P("This Analysis uses the Fast Gradient Method to generate adversarial images.", style=ts))
        result.append(html.P("One image is selected across each class (max 10), an an adversarial example is "
                             "generated for each.", style=ts))
        result.append(html.Br())
        result.append(html.B("For this analysis, " + str(total_images) + " images were evenly selected across " + \
                  str(total_classes) + " classes.", style=ts))

        cats = ["Initial Image", "Initial Prediction", "Perturbation", "Adversarial Image", "Final Prediction"]
        table_header = [html.Thead(html.Tr([html.Th(i) for i in cats]))]
        data_rows = []
        for target_class in adv_res:
            og_img = self.to_display_image(adv_res[target_class]["image"].copy())
            adv_img = self.to_display_image(adv_res[target_class]["adversarial"].copy())
            pert_dis = self.get_diff(adv_res[target_class]["image"].copy(), adv_res[target_class]["adversarial"].copy())
            initial_pred = target_class
            final_pred = adv_res[target_class]["final_prediction"]
            data_rows.append(html.Tr([html.Td(og_img), html.Td(html.B(self.output_features[initial_pred])),
                                      html.Td(pert_dis), html.Td(adv_img), html.Td(html.B(self.output_features[final_pred]))]))
        result.append(html.Br())
        width = 70
        small_width = str((100-width)/2) + "%"
        result.append(html.Div(style={"display": "inline-block", "width": small_width}))
        table = dbc.Table(table_header + [html.Tbody(data_rows)], striped=True, bordered=True)
        result.append(html.Div([table], style={"display": "inline-block", "width": str(width) + "%"}))
        result.append(html.Div(style={"display": "inline-block", "width": small_width}))
        return html.Div(result)
