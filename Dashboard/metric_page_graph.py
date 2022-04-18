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
    fig = px.line( pd.DataFrame(d), x="x", y="value", color="metric", markers ="True" )
    c = dcc.Graph(figure=fig)

    return  html.Div([
    html.P(""),
    html.P(""),
    html.P(""),
    html.P(""),
    html.P(""),
    
     html.Div( 
        
        html.Div( 
            [
                html.P( "select metric group"),
                dcc.Dropdown( groups,  id='select_group'),
                html.P( ""),
                html.P( "select metric"),
                html.Div(id='select_metric_cnt'),
            ], 
        style={ "margin":"20px",
                }),

        style = {"background-color":"Azure",
                "border-width": "thin",
                "border-color":"Blue",
                "border-style":"solid",
                "border-radius": "10px",
                }
    ),
    
    
    html.Hr(),
    html.Div( 
        html.Div(id='graph_cnt', children = [c],
         style={ "margin":"2",
                }), 
        style = {"background-color":"Azure",
                "border-width": "thin",
                "border-color":"LightGray",
                "border-style":"solid",
                "border-radius": "3px",
                "margin":"4"
                })
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
        return dcc.Dropdown( [],  id='select_metrics'),
    # print('value :',value )
    
    metrics = []
    for m in redisUtil.info["metric_info"][value]:
        if m == "meta": 
            continue
        if redisUtil.info["metric_info"][value][m]["type"] in ["numeric"]:
            metrics.append(m)
    # print(metrics)
    return html.Div( [
        dcc.Dropdown( metrics,  id='select_metrics'),
         dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )])


# callback for group combo
#
#

# @app.callback(
#     Output('graph_cnt', 'children'),
#     Input('interval-component', 'n_intervals'),
#     State('select_metrics', 'value'),
#     State('select_group', 'value'),
#     State('graph_cnt', 'children')
# )
# def update_graph_timer(n,metric,group, old):
#     if not metric or not group:
#         return old
#     if redisUtil.subscribers["metric_graph"]:
#         redisUtil.subscribers["metric_graph"] = False
#         print("update happend")
#         return update_graph(metric,group)
#     else:
#         print("update ignored")
#         return old


    

@app.callback(
    Output('graph_cnt', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('select_metrics', 'value'),
    Input('select_group', 'value'),
    State('graph_cnt', 'children')
)



def update_graph(n,metric,group, old):
    
    ctx = dash.callback_context
    print( ctx.triggered)
    
    if 'prop_id' in ctx.triggered and ctx.triggered['prop_id'] == 'interval-component.n_intervals':
        if redisUtil.subscribers["metric_graph"]:
            print("new data")
            redisUtil.subscribers["metric_graph"] = False
        else:
            print("ignore timer")
            return old


    d = {"x":[], "value":[],"tag":[], "metric":[]}
    if not metric or not group:
        fig = px.line( pd.DataFrame(d), x="x", y="value", color="metric" )
        return dcc.Graph(figure=fig)
    
    for i,data in enumerate(redisUtil.values["metric_values"]):
        d["x"].append(i+1)
        d["value"].append(data[group][metric])
        d["tag"].append(data["metadata"]["tag"])
        d["metric"].append( f"{group} : {metric}")

    df = pd.DataFrame(d)
    # fig = px.line(df, x="x", y="value", color="metric" )
    # fig.update_traces(textposition="bottom right")
    
    
    fig = go.Figure(data=[go.Scatter(x=d["x"], y=d["value"]) ])

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

        
    
