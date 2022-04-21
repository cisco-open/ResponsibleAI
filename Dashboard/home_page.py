import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import logging
logger = logging.getLogger(__name__)
from server import redisUtil
import dash_daq as daq
# import dash_trich_components as dtc
import numpy as np



def get_card( t1, t2, t3, ic, n , id, c):
    # c = "lightcyan"
    c= "lightgoldenrodyellow"
    return dbc.Card(
    [
        # html.I(className=ic, style={}),
        # dbc.CardImg(src="fa-solid fa-list-check", top=True),
        dbc.CardBody(
            [
                html.H4(t1, className="card-title"),
                daq.Gauge(
                            color={"gradient":True,"ranges":{"red":[0,4],"yellow":[4,8],"green":[8,10]}},
                            value=100*np.mean(n),
                            label='',
                            max=100,
                            min=0,
                            size=150
                        )
            ]
        ),
        
        html.Hr(),
        html.P ( "%d of total %d certificates passed"%(
                    np.sum(n), len(n) ) )
    ],
    style={"width": "20rem",
    "background-color":"snow",
    "margin":"20px",
    "border-radius":"10px"
    },
    )


def get_home_page():


    certs = redisUtil.get_certificate_info()
    explain = []
    robust = []
    fair = []
    perform = []

    score_explain = []
    score_robust = []
    score_fair = []
    score_perform = []
    cert_values = redisUtil.get_certificate_values()[-1]
    
    for c in certs:
         
        for t in certs[c]["tags"]:
            if "explain" in t.lower():
                explain.append(c)
                score_explain.append( 1 if cert_values[c]["value"] else 0)
            if "robust" in t.lower():
                robust.append(c)
                score_robust.append( 1 if cert_values[c]["value"] else 0)

            if "perform" in t.lower():
                perform.append(c)
                score_perform.append( 1 if cert_values[c]["value"] else 0)

            if "fair" in t.lower():
                fair.append(c)
                score_fair.append( 1 if cert_values[c]["value"] else 0)
    
   

    gauges  =[dbc.Row(
             [
                get_card("Explainability", "success rate", "details", "fa-solid fa-list-check",  score_explain,"c1", "baige"),
                get_card("Robustness", "success rate", "details", "fa-solid fa-list-check", score_robust,"c2", "lightgoldenrodyellow"),

             ] )
             ,
        dbc.Row(
             [
                get_card("Performance", "success rate", "details", "fa-solid fa-list-check", score_perform,"c3","lightcyan"),
                get_card("Fairness", "success rate", "details", "fa-solid fa-list-check",  score_fair,"c4","peachpuff"),

             ] )]
    # return html.P("This is the content of the home page!")
    from certificate_page import generate_cert_table

    return html.Div([
        
        dbc.Row([
            dbc.Col(gauges), 
            dbc.Col(
                html.Div( generate_cert_table(-1,False) , style = {"border-color":"silver", "border-radius":"1px", "border-style":"solid","margin-top":"20px","width":"300px"} ) )
        ])
        

    ])
