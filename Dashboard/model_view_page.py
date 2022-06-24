import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import logging
logger = logging.getLogger(__name__)
import urllib 
from dash import dash_table
import pandas as pd
 
import json
import sklearn
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import base64
import plotly.graph_objs as go
# from dtreeviz.trees import dtreeviz # remember to load the package
import graphviz
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

import matplotlib.pyplot as plt
 


import chart_studio
import plotly.tools as tls


def get_mdl_image(nM, nD):
    vs = redisUtil.get_metric_values()
    rf =  pickle.loads(vs[nM]['tree_model_metadata']['estimator_params'][nD].encode('ISO-8859-1') )
    feat_names = vs[nM]['tree_model_metadata']['feature_names']
    
    fig = plt.figure( figsize=[6,4])
    sklearn.tree.plot_tree(rf, filled=True,fontsize=8,feature_names = feat_names  )
    fig.set_size_inches(8,5)
    buf = io.BytesIO() # in-memory files
    
    fig.savefig(buf, format = "png") # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    return html.Img(id='example1',src = "data:image/png;base64,{}".format(data))




def get_mdl_text(nM, nD):
    vs = redisUtil.get_metric_values()
    rf =  pickle.loads(vs[nM]['tree_model_metadata']['estimator_params'][nD].encode('ISO-8859-1') )
    feat_names = vs[nM]['tree_model_metadata']['feature_names']
    
    
    text_representation = sklearn.tree.export_text(rf, feature_names = feat_names)
    # print( rf.__dict__.keys())
    return  html.Div([
    dcc.Textarea(
        id='textarea-example',
        value=text_representation,
        style={'width': '100%', 'height': 700, 'padding':'25px'},
    ),
    html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line' })

    ])



def get_form():

    ops = []
    values = redisUtil.get_metric_values() 
    for i,m in enumerate(values):
        ops.append( {"label":  m["metadata"]["date"]+  " - " +  m["metadata"]["tag"] , 
                     "value":i})

     
    dropdown = html.Div(
        [
            dbc.Label("Select Measurement", html_for="dropdown"),
            dcc.Dropdown(
                id="measurement_selector",
                options= ops[::-1] ,
                value= len(values)-1
            ),
        ],
        className="mb-3",
    )

    vs = redisUtil.get_metric_values()
    dropdown_tree = html.Div(
        [
            dbc.Label("Select Decision Tree", html_for="dropdown"),
            dcc.Dropdown(
                id="tree_selector",
                options= list(range(len(vs[-1]['tree_model_metadata']['estimator_params']))) ,
                value= 0
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
                options=[
                    {"label": "Textual", "value": 1},
                    {"label": "Graphical", "value": 2},
                   
                ],
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
    
    return html.Div( [
        dbc.Col( 
            html.Div( 
        
        html.Div(get_form(),
        style={ "margin":"5px",
                }),
        style = {"background-color":"rgb(198,216,233)",
                "border-width": "thin",
                "border-color":"silver",
                "border-style":"solid",
                "border-radius": "10px",
                "padding":"10px"
                }
                ),
        ) ,
        html.Div( id = "model_view") 
    ])
    return get_mdl_image(0)
    return get_mdl_text(0)
     
@app.callback(
    Output('model_view', 'children'),
    Input('measurement_selector', 'value'),
    Input('tree_selector', 'value'),
    Input('visual_type', 'value'),
    
)
def update_model_view(m,t, v):
    # print(v)
    if v==2:
        return get_mdl_image(m,t)
    
    if v==1:
        return get_mdl_text(m,t)
    