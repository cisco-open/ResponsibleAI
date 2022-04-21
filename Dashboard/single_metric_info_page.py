import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from server import app, redisUtil
from utils import process_cell
import logging
logger = logging.getLogger(__name__)

def generate_table(group_name, metric_name):
    
    # print(redisUtil.get_metric_info().keys())
    gr = redisUtil.get_metric_info()[group_name]
    # print(gr.keys())
    rows = []
    rows.append( html.Tr([  html.Th("Group Name"),  html.Th( group_name)]))
    rows.append( html.Tr([  html.Th("Group Tag"),  html.Th(gr["meta"]["tags"] )]))
    rows.append( html.Tr([  html.Th("Complexity"),  html.Th(gr["meta"]["complexity_class"] )]))
    rows.append( html.Tr([  html.Th("Depenencies"),  html.Th(gr["meta"]["dependency_list"] )]))
    # print( gr[metric_name].keys())
    for k,v in gr[metric_name].items():
        rows.append( html.Tr([  html.Th(k),  html.Th(v )]))


     
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
     



def get_single_model_info_page(group_name, metric_name):
    
    
    return html.Div([
    html.H4(children='Properties of the AI system'),
    generate_table( group_name, metric_name)
    ])

    