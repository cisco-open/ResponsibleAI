import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from server import app, redisUtil
import pprint
pp = pprint.PrettyPrinter(indent=4)
import json

def dic2tbl(d):
    
     
    return dbc.Table(
        # className="model-info-table",
        children = [
        html.Thead(
            html.Tr([  html.Th(x) for x in d.keys()  ])
        ),
        
        html.Tbody(
            html.Tr([  html.Td( process(x) ) for x in d.values()  ])
          )
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


def process(v):
     
    if type(v) in (tuple, list):
        # return pp.pformat(v)
        return [ dbc.Row(dbc.Col(html.Div(str(x)))) for x in v] 
        return ("\n".join(v))
    if isinstance(v,dict):
        return dic2tbl(v)
        # return pp.pformat(v)
        return json.dumps(v, indent=4)
    return str(v)

def generate_table():
    
    rows = []
    for k,v  in  redisUtil.info["model_info" ].items():
        rows.append ( html.Tr( 
            [ html.Th(k), html.Td( process(v)) ]
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

    