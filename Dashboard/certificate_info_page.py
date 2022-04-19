import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from server import app, redisUtil
import pprint
pp = pprint.PrettyPrinter(indent=4)
import json


def get_accordion():

    items = []

    certs = redisUtil.info["certificate_info" ].values()
    
    for c in certs:
        
        # print( c.keys())    
         
        print(c)
        detail = dbc.Table(
            children = [
                html.Thead(
                     html.Tr([  html.Th("Description") ,
                                html.Th('Tags'),
                                html.Th("Level"),
                                html.Th("Condition")])
                                                    ),

                html.Tbody(  
                    html.Tr([  html.Td( process(c["description"])) ,
                                html.Td(process(c['tags'])),
                                html.Td(process(c["level"])),
                                html.Td(process(c["condition"]))])
                                                    ) 
            ] 
        ,
        bordered=True,
        striped=True

        )
        
        items.append(
            dbc.AccordionItem(
                children=detail,
                title=c['display_name'],
                item_id=c['display_name']
            ),
        )
    acc = dbc.Accordion(
        items,
        active_item= items[0].item_id,
        start_collapsed=False,
        always_open=True
        
    )
    return acc

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
     
    if isinstance(v,dict):
        return dic2tbl(v)
    if type(v) in (tuple, list):
        print("list found", v)
        # return pp.pformat(v)
        return [ dbc.Row(dbc.Col(html.Div(str(x)))) for x in v] 
        return ( "\n".join( [ str (x) for x in v] ))
    if isinstance(v,dict):
        return pp.pformat(v)
        return json.dumps(v, indent=4)
    return str(v)
import pandas as pd
def generate_table():
    
    # df = pd.DataFrame(redisUtil.info["certificate_info" ])
    # return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    rows = []
    for k,v  in  redisUtil.info["certificate_info" ].items():
        
        rows.append ( html.Tr( 
            [ html.Td([process(x)])   for x in v.values()  ]
        ))
        
        # rows.append( html.Thead(
        # rows.append ( html.Tr( 
        #     [ html.Td(k), html.Td( process(v)) ]
        # ))
        
    
    return dbc.Table(
        # className="model-info-table",
        children = [
        html.Thead(
            # html.Tr([  html.Th("Certificate")   ])
            html.Tr([ html.Th(x)   for x in next( iter(redisUtil.info["certificate_info" ].values()) )  ])
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
    # generate_table( )
    get_accordion()
    ])

 