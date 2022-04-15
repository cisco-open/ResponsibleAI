import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

def get_home_page():
    return html.P("This is the content of the home page!")
