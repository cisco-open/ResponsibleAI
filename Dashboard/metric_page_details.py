import logging
logger = logging.getLogger(__name__)

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import pandas as pd
import urllib
from dash import dcc
import plotly.express as px
import plotly.graph_objs as go
 

from utils import process_cell


def get_accordion(id):

    items = []

    values = redisUtil.get_metric_values()[id]
    for group in values:
        
        
        rows = []
        for k,v  in  values[group].items():
            
            if isinstance(v,dict):
                vs = []
                for ki,vi in v.items():
                    vs.append( html.Div( process_cell( {ki:vi}) , style={"max-width": "1200px"} ) )
            else:
                vs = process_cell(v, list_vertical =   isinstance(v,dict))
            
             
            qs = urllib.parse.urlencode( {"g":group, "m":k})
            ico = html.I( n_clicks=0, className='fa-solid fa-info')
            btn = dbc.Button(
                    html.Span([ico]) , href="/single_metric_info/?" + qs, style = { "width":"3px",  "margin-right":"5px"}, outline=True, color="light"
                           ,className="me-1" )
            rows.append ( html.Tr( 
                [ html.Td(  html.Div ( [  btn, k]  ) ), html.Td( vs) ]
            ))

        detail = dbc.Table(
                            children = [
                                        html.Thead(
                                                    html.Tr([  html.Th("Metric Name") , html.Th("Metric Value") ])
                                                    ),

                                        html.Tbody( rows )
                             ],
        bordered=True,
        striped=True ,
        responsive=True,
        size = 'sm'
        
        )
        items.append(
            dbc.AccordionItem(
                children=detail,
                title= redisUtil.get_metric_info()[group]["meta"]["display_name"],
                item_id=group
            ),
        )
    acc = dbc.Accordion(
        items,
        active_item= items[0].item_id,
        # start_collapsed=False,
        # always_open=True
        flush=True,
    )
    return acc



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


    return dbc.Form([
        dropdown

    ])


def get_metric_page_details():
    
     
    return  html.Div([
    html.P(""),
    html.P(""),
    html.P(""),
    html.P(""),
    html.P(""),
    
     
     
    # html.Hr(),
    html.Div( 
        
        html.Div(get_form(),
        style={ "margin":"20px",
                }),
        style = {"background-color":"Azure",
                "border-width": "thin",
                "border-color":"silver",
                "border-style":"solid",
                "border-radius": "10px",
                "padding":"10px"
                }
    ),
    
    html.Hr(),
    
    html.Div( 
        html.Div( id = "measure_accordion", 
                # style= { "margin":"1px",
                # "border-width": "thin",
                # "border-color":"Blue",
                # "border-style":"solid",
                # "border-radius": "10px",
                #     }
                    ) ,

        # style = {"background-color":"AliceBlue",
        #         "border-width": "thin",
        #         "border-color":"Blue",
        #         "border-style":"solid",}
    )

    ])
 

 
@app.callback(
    Output('measure_accordion', 'children'),
    Input('measurement_selector', 'value'),
    
)
def update_metrics(value):

    return get_accordion(value)
