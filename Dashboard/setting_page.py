import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import logging
logger = logging.getLogger(__name__)
import urllib 
from dash import dash_table
import pandas as pd

tbl_styling = { 
    'style_data':{
        'color': 'black',
        'backgroundColor': 'white',
        'textAlign': 'left',
        "whiteSpace": "pre-line"
    },
    'style_data_conditional':[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
    ],
    'style_header':{
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold',
        'textAlign': 'center'
    }}
 
    
 

  
def get_setting_page():
    
    qs = urllib.parse.urlencode( {"g":"metadata", "m":"date"})
    
    return html.Div( [

        html.H3("Display Settings") ,
        html.Div(id = "dummy_setting") ,
        html.Hr(),
        dbc.Form ([
                    html.Div(
                            [
                                dbc.FormText("Precision for floating points data"),
                                dbc.Input(id="input_precision", type="number", min=0, max=6, step=1, value = redisUtil._precision,
                                style={'width': '200px'}),
                            ],
                            id="styled-numeric-input",) ,
                    
                    html.Div(
                            [
                                dbc.FormText("Maximum text length"),
                                dbc.Input(id="input_maxlen", type="number", min=1, max=500, step=1, value = redisUtil._maxlen,
                                style={'width': '200px'}),
                            ],
                            id="styled-numeric-input",),
                    html.Div(
                        dbc.Button( "Apply", id="apply_setting", href="/single_metric_info/?" + qs)
                    )
                    
        ]),
        
        html.Div( id = "setting_div")
    ])
#Input("input_precision", "value"),
@app.callback(
    Output("dummy_setting", "children"),
       
    
    [
        Input("input_maxlen", "value"),
        Input("input_precision", "value"),
        
    ],
)
def on_form_change( maxlen, precision ):
     
    if maxlen is not None:
        redisUtil._maxlen = maxlen
    if precision:
        redisUtil.reformat( precision )
    return []
