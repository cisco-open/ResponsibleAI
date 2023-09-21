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
from dash import Input, Output, dcc, html
from .server import app, dbUtils
import sklearn
import pickle
import io
import base64
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def get_mdl_image(nM, nD):
    dataset = dbUtils.get_current_dataset()
    vs = dbUtils.get_metric_values()
    rf = pickle.loads(vs[nM][dataset]['tree_model_metadata']['estimator_params'][nD].encode('ISO-8859-1'))
    if rf.max_depth and rf.max_depth > 5:
        return html.Div([
            html.Br(),
            html.H5("The graphical view is disabled due to size limitation")],)
    feat_names = vs[nM][dataset]['tree_model_metadata']['feature_names']
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=[6, 4])
    sklearn.tree.plot_tree(rf, filled=True, fontsize=8, feature_names=feat_names)
    fig.set_size_inches(8, 5)
    buf = io.BytesIO()  # in-memory files

    fig.savefig(buf, format="png")  # save to the above file object

    data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
    return html.Img(id='example1', src="data:image/png;base64,{}".format(data))


def get_mdl_text(nM, nD):
    dataset = dbUtils.get_current_dataset()
    vs = dbUtils.get_metric_values()
    rf = pickle.loads(vs[nM][dataset]['tree_model_metadata']['estimator_params'][nD].encode('ISO-8859-1'))
    feat_names = vs[nM][dataset]['tree_model_metadata']['feature_names']
    text_representation = sklearn.tree.export_text(rf, feature_names=feat_names)
    return html.Div([
        dcc.Textarea(id='textarea-example', value=text_representation,
                     style={'width': '100%', 'height': 700, 'padding': '25px'}),
        html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})
    ])


def get_form():
    ops = []
    dataset = dbUtils.get_current_dataset()
    values = dbUtils.get_metric_values()
    for i, m in enumerate(values):
        ops.append({"label": m[dataset]["metadata"]["date"] + " - " + m[dataset]["metadata"]["tag"], "value": i})

    dropdown = html.Div(
        [
            dbc.Label("Select Measurement", html_for="dropdown"),
            dcc.Dropdown(id="measurement_selector", options=ops[::-1], value=len(values) - 1),
        ],
        className="mb-3",
    )

    vs = dbUtils.get_metric_values()
    dropdown_tree = html.Div(
        [
            dbc.Label("Select Decision Tree", html_for="dropdown"),
            dcc.Dropdown(
                id="tree_selector",
                options=list(range(len(vs[-1][dataset].get('tree_model_metadata', {}).get('estimator_params', [])))),
                value=0
            ),
        ],
        className="mb-3",
    )

    radios_input = dbc.Row(
        [
            dbc.Label("Select Visualization Type", html_for="example-radios-row", width=2),
            dbc.Col(
                dbc.RadioItems(
                    id="visual_type",
                    options=[{"label": "Textual", "value": 1}, {"label": "Graphical", "value": 2}],
                    value=2
                ),
            ),
        ],
        className="mb-3",
    )

    return dbc.Form([
        dropdown,
        dropdown_tree,
        radios_input
    ])


def get_model_view_page():
    return html.Div([
        dbc.Col(
            html.Div(
                html.Div(get_form(),
                         style={"margin": "5px"}),
                style={"background-color": "rgb(198,216,233)",
                       "border-width": "thin",
                       "border-color": "silver",
                       "border-style": "solid",
                       "border-radius": "10px",
                       "padding": "10px"}
            ),
        ),
        html.Div(id="model_view")
    ])


@app.callback(
    Output('model_view', 'children'),
    Input('measurement_selector', 'value'),
    Input('tree_selector', 'value'),
    Input('visual_type', 'value'),
)
def update_model_view(m, t, v):
    if v == 2:
        return get_mdl_image(m, t)

    if v == 1:
        return get_mdl_text(m, t)
