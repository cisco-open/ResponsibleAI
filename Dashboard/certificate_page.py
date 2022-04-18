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



def get_form():

    ops = []
    values = redisUtil.values["metric_values"]
    for i,m in enumerate(values):
        ops.append( {"label":  m["metadata"]["date"]+  " - " +  m["metadata"]["tag"] , 
                     "value":i})

    
    dropdown = html.Div(
        [
            dbc.Label("Select Measurement", html_for="dropdown"),
            dcc.Dropdown(
                id="measurement_selector_cert",
                options= ops[::-1] ,
                value= len(values)-1
            ),
        ],
        className="mb-3",
    )


    return dbc.Form([
        dropdown

    ])



def generate_table(id):
    
    rows = []
    for k,v  in  redisUtil.values["certificate_values" ][id].items():
        rows.append ( html.Tr( 
            [ html.Td(k[:-4]), html.Td( v['explanation'].ljust(20) ), html.Td( "Passed" if v['value']==True else "Failed") ]
        ))
    return dbc.Table(
        # className="cert_table",
        children = [
        html.Thead(
            html.Tr([  html.Th("Cetrificate") , html.Th("Explanation"), html.Th("Status") ])
        ),
        
        html.Tbody(  rows)
        ], striped=True
        )
     



def get_certificate_page():
    
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
                "border-color":"Blue",
                "border-style":"solid",
                "border-radius": "10px",
                }
    ),
    
    html.Hr(),
    
    html.Div( 
        html.Div( id = "certificate_pane", 
                style= { "margin":"30px",
               
                    }) ,
        style= { "margin":"5",
         "border-width": "thin",
                "border-color":"Blue",
                "border-style":"solid",
                "border-radius": "10px",}
        # style = {"background-color":"AliceBlue",
        #         "border-width": "thin",
        #         "border-color":"Blue",
        #         "border-style":"solid",}
    )

    ])
 
    # return html.Div([
    # html.H4(children='Certificates'),
    # generate_table( )
    # ])

 
@app.callback(
    Output('certificate_pane', 'children'),
    Input('measurement_selector_cert', 'value'),
    
)
def update_metrics(value):

    return generate_table(value)
