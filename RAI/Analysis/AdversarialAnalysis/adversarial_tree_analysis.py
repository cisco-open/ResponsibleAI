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

from RAI.Analysis import Analysis
from RAI.AISystem import AISystem
from art.metrics.verification_decisions_trees import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification.scikitlearn import SklearnClassifier
import os
import numpy as np
from dash import html, dcc
import plotly.graph_objs as go


class AdversarialTreeAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.search_steps = 10
        self.search_steps = 3
        self.distortion_size = 0.3
        self.max_progress_tick = 4

    @classmethod
    def is_compatible(cls, ai_system: AISystem, dataset: str):
        compatible = super().is_compatible(ai_system, dataset)
        model_types = ["ExtraTreesClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]
        compatible = compatible and any(i in str(ai_system.model.agent.__class__) for i in model_types)
        return compatible and str(ai_system.model.agent.__class__).startswith("<class 'sklearn.ensemble.")

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        self.progress_tick()
        data = self.ai_system.get_data(self.dataset)
        classifier = SklearnClassifier(model=self.ai_system.model.agent)
        self.progress_tick()
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        self.progress_tick()
        if data.y.ndim == 1:
            y = np.stack([data.y == 0, data.y == 1], 1)
        else:
            y = data.y

        # Note: This runs slow, to speed it up we can take portion of test set size
        result['adversarial_tree_verification_bound'], result['adversarial_tree_verification_error'] = \
            rt.verify(data.X, y, eps_init=self.distortion_size, nb_search_steps=self.search_steps, max_clique=2, max_level=2)
        self.progress_tick()
        return result

    def to_string(self):
        result = "\n==== Decision Tree Adversarial Analysis ====\n"
        result += "This test uses the Clique Method Robustness Verification method.\n" \
                  "The Adversarial Tree Verification Lower Bound describes the lower bound of " \
                  "minimum L-infinity adversarial distortion averaged over all test examples.\n"
        result += "Adversarial Tree Verification Lower Bound: " + str(self.result['adversarial_tree_verification_bound'])\
                  + '\n'
        result += f"\nAdversarial Tree Verified Error is the upper bound of error under any attacks.\n" \
                  f"Verified Error guarantees that within a L-infinity distortion norm of {str(self.distortion_size)}" \
                  f", that no attacks can achieve over X% error on test sets.\n"
        result += "Adversarial Tree Verified Error: " + str(self.result['adversarial_tree_verification_error']) + "\n"
        return result

    def to_html(self):
        result = []
        text_style = {"text-align": "center", "display": "block"}
        bound = self.result['adversarial_tree_verification_bound']
        error = self.result['adversarial_tree_verification_error']
        result.append(html.H1("Decision Tree Adversarial Analysis", style=text_style))
        result.append(html.P("This test uses the Clique Method Robustness Verification method.", style=text_style))
        result.append(html.Br())
        result.append(html.B("Adversarial Tree Verification Lower Bound: " + str(bound)))
        result.append(html.Br())
        result.append(html.P("The Adversarial Tree Verification Lower Bound describes the lower bound of "
                             "minimum L-infinity adversarial distortion averaged over all test examples."))
        result.append(html.Br())
        result.append(html.B("Adversarial Tree Verified Error: " + str(error)))
        result.append(html.Br())
        result.append(html.P(
            f"Adversarial Tree Verified Error is the upper bound of error under any attacks. "
            f"Verified Error guarantees that within a L-infinity distortion norm of {str(self.distortion_size)}"
            ", that no attacks can achieve over X% error on test sets."))

        fig = go.Figure([go.Bar(x=["Lower Bound", "Verified Error"], y=[bound, error])])
        fig.update_layout(title={'text': "Adversarial Tree Results", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
        graph_max = max(bound, error) * 1.2
        fig.update_layout(yaxis_range=[0, graph_max])
        fig_graph = html.Div(dcc.Graph(figure=fig), style={"display": "block", "margin": "0 auto 0 auto", "width": "60%"})
        result.append(fig_graph)
        return html.Div(result)
