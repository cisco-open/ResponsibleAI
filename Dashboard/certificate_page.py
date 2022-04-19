import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from server import app, redisUtil
from utils import process_cell 

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
        
        if v['value']:
            status = html.Div( [
                "Passed ", html.I( className = "fa-solid fa-check", style={"width":"30px","height":"30px", "margin-left":"10px", "color":"green"})
            ])
        else:
            status = html.Div( [
                 "Failed" , html.I( className = "fa-solid fa-xmark", style={"width":"30px","height":"30px","margin-left":"25px", "color":"red"})
            ])
        rows.append ( html.Tr( 
            [ html.Td(k[:-4]), html.Td( v['explanation'].ljust(20) ), html.Td(status) ]
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
