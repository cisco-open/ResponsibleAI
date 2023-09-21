# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import logging
import dash_bootstrap_components as dbc
from dash import html
from .server import dbUtils
from .utils import process_cell

logger = logging.getLogger(__name__)


def get_accordion():
    items = []
    certs = dbUtils.get_certificate_info().values()

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
                title=c['display_name'].title(),
                item_id=c['display_name'],
            ))
    if len(certs) > 0:
        acc = dbc.Accordion(items, active_item=items[0].item_id)
    else:
        acc = dbc.Accordion(items, active_item=None)
    return acc


def generate_table():
    rows = []
    for k, v in dbUtils.get_certificate_info().items():
        rows.append(html.Tr([html.Td([process_cell(x)]) for x in v.values()]))

    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th(x) for x in next(iter(dbUtils.get_certificate_info().values()))])),
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
