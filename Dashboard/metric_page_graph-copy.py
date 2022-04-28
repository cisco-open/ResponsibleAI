import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from dash import dcc
import plotly.express as px
import plotly.graph_objs as go

# layout = None

def get_metric_page_graph():
     
    groups = []
    
    for g in redisUtil.get_metric_info():
        groups.append(g)


    d = {"x":[], "value":[],"tag":[], "metric":[]}
    fig = px.line( pd.DataFrame(d), x="x", y="value", color="metric", markers ="True" )
    c = dcc.Graph(figure=fig)
    v = None
    if groups:
        v = groups[0]
    layout =   html.Div([
    
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
         ),

        html.P(""),
        html.P(""),
        html.P(""),
        html.P(""),
        html.P(""),
    
        html.Div( 
        
            html.Div( [
                    dbc.Label("select metric group", html_for="select_group"),
                    # html.P( "select metric group"),
                    dcc.Dropdown( groups,  id='select_group',value=v, persistence=True, persistence_type='session'),
                    html.P( ""),
                    dbc.Label("select metric", html_for="select_metric_cnt"),
                    # html.P( "select metric"),
                    html.Div(id='select_metric_cnt', 
                    children=html.Div(id="select_metrics")),
                ], 
                style={ "margin":"20px",
                    }),

            # style = {"background-color":"Azure",
            #         "border-width": "thin",
            #         "border-color":"Blue",
            #         "border-style":"solid",
            #         "border-radius": "10px",
            #         }
            style = {"background-color":"rgb(198,216,233)",
                "border-width": "thin",
                "border-color":"silver",
                "border-style":"solid",
                "border-radius": "10px",
                "padding":"10px",
                 "border-radius": "3px",
                }
            ),
    
    
            html.Br(),
            html.Div( 
                html.Div(id='graph_cnt', children = [c],
                style={ "margin":"2",
                        }), 
            style = {"background-color":"rgb(198,216,233)",
                    "border-width": "thin",
                    "border-color":"LightGray",
                    "border-style":"solid",
                    "border-radius": "3px",
                    "margin":"4"
                    })
        ])

    return layout
# callback for group combo
#
#

@app.callback(
    Output('select_metric_cnt', 'children'),
    Input('select_group', 'value'),
    State('select_metric_cnt', 'children')
    
)

def update_metrics(value, children):
     
    if not value:
        # return children
        return dcc.Dropdown( [],  id='select_metrics',persistence=True, persistence_type='session')
        
     
    metrics = []
    for m in redisUtil.get_metric_info()[value]:
        if m == "meta": 
            continue
        if redisUtil.get_metric_info()[value][m]["type"] in ["numeric"]:
            metrics.append(m)
     
    return  dcc.Dropdown( metrics,  id='select_metrics',persistence=True, persistence_type='session') 
  
@app.callback(
    Output('graph_cnt', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('select_metrics', 'value'),
    Input('select_group', 'value'),
    State('graph_cnt', 'children')
)



def update_graph(n,metric,group, old):
    
    ctx = dash.callback_context
     
    
    if 'prop_id' in ctx.triggered and ctx.triggered['prop_id'] == 'interval-component.n_intervals':
        if redisUtil.has_update("metric_graph", reset = True):
            logger.info("new data")
            redisUtil.subscribers["metric_graph"] = False
        else:
            logger.info("ignore timer")
            return old


    d = {"x":[], "value":[],"tag":[], "metric":[],"text":[]}
    if not metric or not group:
        fig = px.line( pd.DataFrame(d), x="x", y="value", color="metric" )
        return dcc.Graph(figure=fig)
    
    for i,data in enumerate(redisUtil.get_metric_values() ):
        d["x"].append(i+1)
        d["value"].append(data[group][metric])
        d["tag"].append(data["metadata"]["tag"])
        # d["metric"].append( f"{group} : {metric}")
        d["metric"].append( f"{metric}")
        d["text"].append( "%.2f"%data[group][metric])

    df = pd.DataFrame(d)
    # fig = px.line(df, x="x", y="value", color="metric" )
    # fig.update_traces(textposition="bottom right")
    
    
    fig = px.line(df, x="x", y="value", color="metric", markers=True, text="text" ) 
    fig.update_traces(textposition="top center")
    fig.update_layout(
        
        xaxis = dict(
            tickmode = 'array',
            tickvals =  d["x"],
            ticktext = d["tag"]
            
        )
        ,
        legend=dict(
            
            
            title_font_family="Times New Roman",
            font=dict(
                family="Times New Roman",
                size=14,
                color="black"
            ),
            bgcolor="Azure",
            bordercolor="Black",
            borderwidth=1 
             
            )
       

        
        
    )   
    return dcc.Graph(figure=fig ) 
