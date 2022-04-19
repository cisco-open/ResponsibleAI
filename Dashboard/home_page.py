import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import dash_daq as daq
# import dash_trich_components as dtc
 
def get_card( t1, t2, t3, ic, n , id, c):
    # c = "lightcyan"
    # c= "lightgoldenrodyellow"
    return dbc.Card(
    [
        # html.I(className=ic, style={}),
        # dbc.CardImg(src="fa-solid fa-list-check", top=True),
        dbc.CardBody(
            [
                html.H4(t1, className="card-title"),
                
                daq.Gauge(
    color={"gradient":True,"ranges":{"red":[0,4],"yellow":[4,8],"green":[8,10]}},
    value=n,
    label='',
    max=10,
    min=0,
    size=120
)


                # daq.Knob(
                # label=t2,
                # value=n,
                # color={"gradient":True,"ranges":{"green":[0,5],"yellow":[5,9],"red":[9,10]}}
                # )
                ,
                # html.P(
                #     t3,
                #     className="card-text",
                # ),
                # dbc.Button("Go somewhere", color="primary"),
            ]
        ),
    ],
    style={"width": "20rem",
    "background-color":c,
    "margin":"20px",
    "border-radius":"10px"},
    )


def get_home_page():



    # return html.P("This is the content of the home page!")
    return html.Div([
        dbc.Row(
             [
                get_card("Explainability", "success rate", "details", "fa-solid fa-list-check", 7,"c1", "baige"),
                get_card("Robustness", "success rate", "details", "fa-solid fa-list-check", 2,"c2", "lightgoldenrodyellow"),

             ] )
             ,
        dbc.Row(
             [
                get_card("Performance", "success rate", "details", "fa-solid fa-list-check", 7,"c3","lightcyan"),
                get_card("Fairness", "success rate", "details", "fa-solid fa-list-check", 2,"c4","peachpuff"),

             ] )

    ])
