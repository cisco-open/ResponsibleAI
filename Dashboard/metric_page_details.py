import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import pandas as pd

from dash import dcc
import plotly.express as px
import plotly.graph_objs as go
 
def get_metric_page_details():
    
     
    return  html.Div([
    html.P( "metric detials"),
     
    html.Hr(),

    ])
 