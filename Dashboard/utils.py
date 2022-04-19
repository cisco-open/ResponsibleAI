from dash import html
import dash_bootstrap_components as dbc


def Iconify(txt, icon, margin="10px", style=None):

    if style is None:
        style={ "margin-left":margin }

    return html.Div([txt,
                    html.I(className=icon, style=style)])


def dic2tbl(d,list_vertical = True):
    
     
    return dbc.Table(
        # className="model-info-table",
        children = [
        html.Thead(
            html.Tr([  html.Th(x) for x in d.keys()  ])
        ),
        
        html.Tbody(
            html.Tr([  html.Td( process_cell(x, list_vertical) ) for x in d.values()  ])
          )
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={ "while-space":"pre",
                "padding-top": "12px",
                "padding-bottom": "12px"}
        )



def process_cell(v, list_vertical = True):
     
    if type(v) in (tuple, list):
        # return pp.pformat(v)
        if list_vertical:
            return [ dbc.Row(dbc.Col(html.Div(str(x)))) for x in v] 
        else:
            return str(v)   
    
    if isinstance(v,dict):
        return dic2tbl(v,list_vertical)
        # return pp.pformat(v)
        return json.dumps(v, indent=4)
    return str(v)