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
from .utils import process_cell
logger = logging.getLogger(__name__)


def generate_table():
    rows = []
    for k, v in dbUtils.get_project_info().items():
        rows.append(html.Tr(
            [html.Th(k), html.Td(process_cell(v))]
        ))
    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th("Property Name"), html.Th("Property Value")])),
            html.Tbody(rows)
        ],
        bordered=False,
        striped=True,
        style={}
    )


def get_model_info_page():
    return html.Div([
        html.H4("Project Info"),
        generate_table()
    ])
