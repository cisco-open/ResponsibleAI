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
import os
from dash import html, dcc
import plotly.graph_objs as go


class FairnessAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None  # calculate result
        self.values = ai_system.get_metric_values()[dataset]['group_fairness']
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.max_progress_tick = 1

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {'statistical_parity_difference': 0.1 >= self.values['statistical_parity_difference'] >= -0.1,
                  'equal_opportunity_difference': 0.1 >= self.values['equal_opportunity_difference'] >= -0.1,
                  'average_odds_difference': 0.1 >= self.values['average_odds_difference'] >= -0.1,
                  'disparate_impact_ratio': 1.25 >= self.values['disparate_impact_ratio'] >= 0.8}
        self.progress_tick()
        return result

    def to_string(self):
        failures = 0
        total = 0
        for val in self.result:
            total += 1
            failures += not self.result[val]
        result = "==== Group Fairness Analysis Results ====\n"
        result += str(total - failures) + " of " + str(total) + " tests passed.\n"
        names = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_odds_difference', 'disparate_impact_ratio']
        conditions = ['between 0.1 and -0.1' for _ in range(3)]
        conditions.append('between 1.25 and 0.8')
        metric_info = self.ai_system.get_metric_info()['group_fairness']
        for i in range(len(names)):
            result += '\n' + metric_info[names[i]]['display_name'] + ' Test:\n'
            result += 'This metric is ' + metric_info[names[i]]['explanation'] + '\n'
            result += "It's value of " + str(self.values[names[i]]) + " is "
            if self.result[names[i]]:
                result += "between " + conditions[i] + " indicating that there is fairness.\n"
            else:
                result += "not between " + conditions[i] + " indicating that there is unfairness.\n"
        return result

    def to_html(self):
        text_style = {"text-align": "center", "display": "block"}
        failures = 0
        total = 0
        for val in self.result:
            total += 1
            failures += not self.result[val]
        result = []
        result.append(html.H1("Group Fairness Analysis Results", style=text_style))
        result.append(html.H2(str(total - failures) + " of " + str(total) + " tests passed.", style=text_style))
        names = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_odds_difference', 'disparate_impact_ratio']
        conditions = ['between 0.1 and -0.1' for _ in range(3)]
        conditions_num = [[0.1, -0.1] for _ in range(3)]
        conditions.append('between 1.25 and 0.8')
        conditions_num.append([1.25, 0.8])
        metric_info = self.ai_system.get_metric_info()['group_fairness']
        for i in range(len(names)):
            result.append(html.Br())
            result.append(html.Br())
            result.append(html.H3(metric_info[names[i]]['display_name'] + ' Test:', style=text_style))
            # https://stackoverflow.com/questions/59120877/how-to-create-a-bar-chart-with-a-mean-line-in-the-dash-app
            graph_range = max(max(abs(self.values[names[i]]), abs(conditions_num[i][0])), abs(conditions_num[i][1]))
            graph_range *= 1.2
            layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0))
            fig = go.Figure([go.Bar(x=[0], y=[self.values[names[i]]])], layout=layout)
            fig.add_shape(go.layout.Shape(
                type="line", x0=-0.5, y0=conditions_num[i][0], x1=0.5, y1=conditions_num[i][0],
                line=dict(color="Orange", width=4, dash="dash")))
            fig.add_shape(go.layout.Shape(
                type="line", x0=-0.5, y0=conditions_num[i][1], x1=0.5, y1=conditions_num[i][1],
                line=dict(color="Orange", width=4, dash="dash")))
            fig.update_layout(yaxis_range=[-graph_range, graph_range])
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(title={'text': metric_info[names[i]]['display_name'], 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
            fig_graph = html.Div(dcc.Graph(figure=fig), style={"display": "block", "margin": "0 auto 0 auto", "width": "60%"})
            result.append(fig_graph)
            result.append(html.Br())
            if self.result[names[i]]:
                result.append(html.B(
                    "It's value of " + str(
                        self.values[names[i]]
                    ) + " is between " + conditions[i] + " indicating that there is fairness.", style=text_style))
            else:
                result.append(html.B(
                    "Its value of " + str(
                        self.values[names[i]]
                    ) + " is not between " + conditions[i] + " indicating that there is unfairness.", style=text_style))
            result.append(html.P('This metric is ' + metric_info[names[i]]['explanation'], style=text_style))
        return html.Div(result)
