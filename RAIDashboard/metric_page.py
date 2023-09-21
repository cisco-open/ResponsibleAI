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
from dash import Input, Output, html
from .server import app
from dash import dcc
from .metric_page_details import get_metric_page_details
from .metric_page_graph import get_metric_page_graph
logger = logging.getLogger(__name__)


def get_metric_page():
    res = html.Div([
        dcc.Tabs(
            id="tabs-with-classes",
            value='details',
            parent_className='custom-tabs',
            className='custom-tabs-container',
            children=[
                dcc.Tab(
                    label='Metric Details',
                    value='details',
                    className='custom-tab',
                    selected_className='custom-tab--selected'
                ),
                dcc.Tab(
                    label='Metric Plots',
                    value='plots',
                    className='custom-tab',
                    selected_className='custom-tab--selected'
                ),
                dcc.Tab(
                    label='Individual Metric Display',
                    value='individual-plots',
                    className='custom-tab',
                    selected_className='custom-tab--selected'
                ),
            ]),
        html.Div(id='tabs-content-classes')])
    return res


@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'details':
        return get_metric_page_details()
    elif tab == 'plots':
        return get_metric_page_graph()
    # elif tab == 'individual-plots':
    #     return get_single_metric_display()
