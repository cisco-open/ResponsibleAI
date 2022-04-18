from dash import html
 


def Iconify(txt, icon, margin="10px", style=None):

    if style is None:
        style={ "margin-left":margin }

    return html.Div([txt,
                    html.I(className=icon, style=style)])
