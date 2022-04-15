import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import pandas as pd

from dash import dcc
import plotly.express as px
import plotly.graph_objs as go


def get_metric_page_graph():
    
    groups = []
    for g in redisUtil.info["metric_info"]:
        
        # print(g)
        groups.append(g)


    d = {"x":[], "value":[],"tag":[], "metric":[]}
    fig = px.line( pd.DataFrame(d), x="x", y="value", color="metric" )
    c = dcc.Graph(figure=fig)

    return  html.Div([
    html.P( "select metric group"),
    
    

    dcc.Dropdown( groups,  id='select_group'),
    html.P( ""),
    html.P( "select metric"),
    
    html.Div(id='select_metric_cnt'),
    html.Hr(),

    html.Div(id='graph_cnt', children = [c], style={"border":"2px black solid", "border-style": "dashed", "bgcolor":"Blue"})

])


# callback for group combo
#
#

@app.callback(
    Output('select_metric_cnt', 'children'),
    Input('select_group', 'value'),
    
)
def update_metrics(value):
    if not value:
        return []
    # print('value :',value )
    
    metrics = []
    for m in redisUtil.info["metric_info"][value]:
        if m == "meta": 
            continue
        if redisUtil.info["metric_info"][value][m]["type"] in ["numeric"]:
            metrics.append(m)
    # print(metrics)
    return dcc.Dropdown( metrics,  id='select_metrics'),


# callback for group combo
#
#

@app.callback(
    Output('graph_cnt', 'children'),
    Input('select_metrics', 'value'),
    State('select_group', 'value')
)
def update_graph(metric,group):
    
    d = {"x":[], "value":[],"tag":[], "metric":[]}
    if not metric or not group:
        fig = px.line( pd.DataFrame(df), x="x", y="value", color="metric" )
        return dcc.Graph(figure=fig)
    
    for i,data in enumerate(redisUtil.values["metric_values"]):
        d["x"].append(i+1)
        d["value"].append(data[group][metric])
        d["tag"].append(data["metadata"]["tag"])
        d["metric"].append( f"{group} : {metric}")

    df = pd.DataFrame(d)
    fig = px.line(df, x="x", y="value", color="metric" )
    fig.update_traces(textposition="bottom right")
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
            family="Courier",
            size=14,
            color="black"
        ),
        bgcolor="Azure",
        bordercolor="Black",
        borderwidth=0)

)
    return dcc.Graph(figure=fig)    
        
    
