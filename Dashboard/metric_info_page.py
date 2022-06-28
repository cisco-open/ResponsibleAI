import logging
import dash_bootstrap_components as dbc
from dash import html
from server import redisUtil
from utils import process_cell
logger = logging.getLogger(__name__)


def get_header_table(group_name, group):
    rows = []
    rows.append(html.Tr(
        [
            html.Td(process_cell(group["meta"]["tags"])),
            html.Td(process_cell(group["meta"]["complexity_class"])),
            html.Td(process_cell(group["meta"]["compatiblity"])),
            html.Td(process_cell(group["meta"]["dependency_list"]))
        ]
    ))

    return dbc.Table(
        children=[
            html.Thead(
                html.Tr([html.Th("Tags"), html.Th("Complexity"), html.Th("Compatiblity"), html.Th("Dependency List")])),
            html.Tbody(rows)
        ],
        bordered=True, striped=True, responsive=True, size='sm')


def get_metric_table(group_name, group):
    rows = []
    keys = [k for k in group if k != "meta"]
    metric_keys = group[keys[0]]

    for k in keys:
        rows.append(html.Tr([html.Td(process_cell(group[k][mk])) for mk in metric_keys]))

    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th(x) for x in metric_keys])),
            html.Tbody(rows)
        ],
        bordered=True, striped=True, responsive=True, size='sm')


def get_accordion():
    items = []

    for group_name, group in redisUtil.get_metric_info().items():
        detail = html.Div([
            get_header_table(group_name, group),
            html.Br(),
            html.P("list of metrics"),
            get_metric_table(group_name, group)
        ])

        display_name = group["meta"]["display_name"]
        items.append(
            dbc.AccordionItem(
                children=detail,
                title=display_name,
                item_id=group_name,
                style={"text-align": "center"}
            ),
        )
    return dbc.Accordion(items, active_item=items[0].item_id)


def get_metric_info_page():
    return html.Div([
        html.H4("Metric Groups"),
        html.Hr(),
        get_accordion()
    ])
