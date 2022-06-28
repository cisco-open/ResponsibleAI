import logging
import dash_bootstrap_components as dbc
from dash import html
from server import redisUtil
from utils import process_cell
logger = logging.getLogger(__name__)


def generate_table():
    rows = []
    for k, v in redisUtil.get_project_info().items():
        rows.append(html.Tr(
            [html.Th(k), html.Td(process_cell(v))]
        ))
    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th("Property Name"), html.Th("Property Value")])),
            html.Tbody(rows)
        ],
        bordered=False,
        striped=True,
        style={}
    )


def get_model_info_page():
    return html.Div([
        generate_table()
    ])
