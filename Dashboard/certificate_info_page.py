import logging
import dash_bootstrap_components as dbc
from dash import html
from server import redisUtil
from utils import process_cell

logger = logging.getLogger(__name__)


def get_accordion():
    items = []
    certs = redisUtil.get_certificate_info().values()

    for c in certs:
        detail = dbc.Table(
            children=[
                html.Thead(
                    html.Tr([html.Th("Description", style={"width": "25%"}),
                             html.Th('Tags'),
                             html.Th("Level"),
                             html.Th("Condition")])
                ),
                html.Tbody(
                    html.Tr([html.Td(process_cell(c["description"]), style={"width": "25%"}),
                             html.Td(process_cell(c['tags'])),
                             html.Td(process_cell(c["level"])),
                             html.Td(process_cell(c["condition"]))])
                )],
            bordered=True,
            striped=True
        )

        items.append(
            dbc.AccordionItem(
                children=detail,
                title=c['display_name'],
                item_id=c['display_name']))
    acc = dbc.Accordion(items, active_item=items[0].item_id)
    return acc


def generate_table():
    rows = []
    for k, v in redisUtil.get_certificate_info().items():
        rows.append(html.Tr([html.Td([process_cell(x)]) for x in v.values()]))

    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th(x) for x in next(iter(redisUtil.get_certificate_info().values()))])),
            html.Tbody(rows)
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={"while-space": "pre", "padding-top": "12px", "padding-bottom": "12px"}
    )


def get_certificate_info_page():
    return html.Div([
        html.H4(children='List of certificates'),
        get_accordion()
    ])
