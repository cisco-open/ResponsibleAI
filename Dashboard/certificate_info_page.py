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
    for k,v  in  redisUtil.info["certificate_info" ].items():
        rows.append ( html.Tr( 
            [ html.Td(k), html.Td( process(v)) ]
        ))
    return dbc.Table(
        # className="model-info-table",
        children = [
        html.Thead(
            html.Tr([  html.Th("Property Name") , html.Th("Property Value") ])
        ),
        
        html.Tbody(  rows)
        ],
         bordered=True,
    # dark=False,
    hover=True,
    responsive=True,
    striped=True,
    style={ "while-space":"pre",
            "padding-top": "12px",
            "padding-bottom": "12px"}
    )
     



def get_certificate_info_page():
    
    
    return html.Div([
    html.H4(children='List of certificates'),
    generate_table( )
    ])

 