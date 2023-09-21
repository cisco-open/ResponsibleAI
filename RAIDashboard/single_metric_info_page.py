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


import logging
import dash_bootstrap_components as dbc
from dash import html
from .server import dbUtils
logger = logging.getLogger(__name__)


def generate_table(group_name, metric_name):
    gr = dbUtils.get_metric_info()[group_name]
    rows = []
    rows.append(html.Tr([html.Th("Group Name"), html.Th(group_name)]))
    rows.append(html.Tr([html.Th("Group Tag"), html.Th(gr["meta"]["tags"])]))
    rows.append(html.Tr([html.Th("Complexity"), html.Th(gr["meta"]["complexity_class"])]))
    rows.append(html.Tr([html.Th("Depenencies"), html.Th(gr["meta"]["dependency_list"])]))
    for k, v in gr[metric_name].items():
        rows.append(html.Tr([html.Th(k), html.Th(v)]))

    return dbc.Table(
        children=[html.Thead(html.Tr([html.Th("Property Name"), html.Th("Property Value")])),
                  html.Tbody(rows)],
        bordered=False,
        striped=True,
        style={}
    )


def get_single_model_info_page(group_name, metric_name):
    return html.Div([
        html.H4(children='Properties of the AI system'),
        generate_table(group_name, metric_name)
    ])
