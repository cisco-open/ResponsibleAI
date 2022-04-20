import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from server import app, redisUtil
import pprint
pp = pprint.PrettyPrinter(indent=4)
import json
from utils import process_cell

def generate_table():
    
    rows = []
    for k,v  in  redisUtil.get_project_info().items():
        rows.append ( html.Tr( 
            [ html.Th(k), html.Td( process_cell(v)) ]
        ))
    return dbc.Table(
        # className="model-info-table",
        children = [
        html.Thead(
            html.Tr([  html.Th("Property Name") , html.Th("Property Value") ])
        ),
        
        html.Tbody(  rows)
        ],
         
        bordered=False,
        striped=True,
        # color = "info",
        style={
        # "border-radius":"40px"
        }
)
     



def get_model_info_page():
    
    
    return html.Div([
    html.H4(children='Properties of the AI system'),
    generate_table( )
    ])

    