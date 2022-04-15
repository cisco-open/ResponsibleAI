import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from server import app, redisUtil
import pprint
pp = pprint.PrettyPrinter(indent=4)
import json

def process(v):
     
    if type(v) in (tuple, list):
        # return pp.pformat(v)
        return ("\n".join(v))
    if isinstance(v,dict):
        return pp.pformat(v)
        return json.dumps(v, indent=4)
    return str(v)

def generate_table():
    
    rows = []
    for k,v  in  redisUtil.info["model_info" ].items():
        rows.append ( html.Tr( 
            [ html.Td(k), html.Td( process(v)) ]
        ))
    return html.Table(
        className="model-info-table",
        children = [
        html.Thead(
            html.Tr([  html.Th("Property Name") , html.Th("Property Value") ])
        ),
        
        html.Tbody(  rows)
        ])
     



def get_model_info_page():
    
    
    return html.Div([
    html.H4(children='Properties of the AI system'),
    generate_table( )
    ])

    