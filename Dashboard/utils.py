import logging
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output
from dash import html
from server import app, redisUtil

logger = logging.getLogger(__name__)
unique_id = [1]
full = {}


def iconify(txt, icon, margin="10px", style=None):
    style = {"margin-right": 15}
    return html.Div([html.I(className=icon, style=style), txt])


def dict_to_table(d, list_vertical=True):
    return dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th(x) for x in d.keys()])),
            html.Tbody(html.Tr([html.Td(process_cell(x, list_vertical)) for x in d.values()]))],
        bordered=True, hover=False, responsive=True, striped=True,
        style={"while-space": "normal", "padding-top": "12px", "padding-bottom": "12px"})


def to_str(v, max_len=None):
    if max_len is None:
        max_len = redisUtil._maxlen
    s = str(v)
    if len(s) > max_len:
        unique_id[0] += 1
        s = s[:max_len]
        full[s] = str(v)
        btn = dbc.Button("...", id={"type": str(v), "index": unique_id[0]},
                         outline=True, color="secondary", className="me-1", size="sm",
                         style={"margin-left": "5px", "width": "28px", "height": "20px",
                                "text-align": "center bottom", "line-height": "5px"})
        return html.Div([s, btn, html.Div(id='dummy_x')])
    return s


@app.callback(
    Output("dummy_x", "children"),
    Input({"type": dash.ALL, "index": dash.ALL}, "n_clicks"))
def show_full_text(x):
    full_text = dash.callback_context.triggered_id['type']
    return dbc.Offcanvas(
        html.P(full_text),
        id="offcanvas", title="Full Content", is_open=True)


def process_cell(v, list_vertical=True):
    if type(v) in (tuple, list):
        # return pp.pformat(v)
        if list_vertical:
            return [dbc.Row(dbc.Col(html.Div(to_str(x)))) for x in v]
        else:
            return to_str(v)

    if isinstance(v, dict):
        return dict_to_table(v, list_vertical)
        # return pp.pformat(v)
        return json.dumps(v, indent=4)
    return to_str(v)
