import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import json
from dash import dcc
import plotly.express as px
import plotly.graph_objs as go
import sd_material_ui
# layout = None


  
def get_trc_data(group, metric):
    
    d = {"x":[], "y":[],"tag":[], "metric":[],"text":[]}
    
     
    for i,data in enumerate(redisUtil.get_metric_values() ):
        d["x"].append(i+1)
        d["y"].append(data[group][metric])
        d["tag"].append(data["metadata"]["tag"])
        # d["metric"].append( f"{group} : {metric}")
        d["metric"].append( f"{metric}")
        d["text"].append( "%.2f"%data[group][metric])

     
    sc_data = { 'mode': 'lines+markers+text' , 
                'name': f"{group}, {metric}", 'orientation': 'v', 'showlegend': True,
                'text':d["text"], 'x': d["x"], 'xaxis': 'x', 'y': d['y'], 'yaxis': 'y' ,'type': 'scatter', 'textposition': 'top center',
                'hovertemplate': 'metric=' + metric + '<br>x=%{x}<br>value=%{y}<br>text=%{text}<extra></extra>' }

    return d["tag"], sc_data




def get_selectors():
    
    groups = []
    for g in redisUtil.get_metric_info():
        groups.append(g)
        
    return html.Div( 
        
        dbc.Form ([

            dbc.Label("select metric group", html_for="select_group"),
            dbc.Row( [
                    dbc.Col([
                            
                            dcc.Dropdown( groups,  id='select_group',value=groups[0] if groups else None, persistence=True, persistence_type='session',placeholder="Select a metric group",),
                            html.P( ""),
                            dbc.Label("select metric", html_for="select_metric_cnt"),
                            # html.P( "select metric"),
                            dcc.Dropdown( [],  id='select_metric_dd',value = None,  placeholder="Select a metric",),
                            
                            # html.Div(id='select_metric_cnt',children=html.Div(id="select_metrics")),
                        ], style={"width":"70%"}),
                        dbc.Col([
                            dbc.Button( "Reset Graph", id="reset_graph",   style = {"margin-left":"20%"}, color="secondary")
                            
                           
                        ],style={"width":"20%"}),
                        
                    ])
                     
            ], style = { "background-color":"rgb(240,250,255)", "width":"100%  ", "border":"solid", "border-color":"silver",  "border-radius":"5px", "padding": "50px"}            
            ), 
        style={ "margin":"2px","margin-bottom":"20px", 
                    }
        )

 
     
def get_graph():
    d = {"x":[], "value":[],"tag":[], "metric":[]}
    fig = px.line( pd.DataFrame(d), x="x", y="value", color="metric", markers ="True" )
    return html.Div( 
                    html.Div(id='graph_cnt', children = [dcc.Graph( figure = fig, id='metric_graph' )],
                    style={ "margin":"1",
                            }), 
            style = {
                # "background-color":"rgb(198,216,233)",
                    "border-width": "thin",
                    "border-color":"LightGray",
                    "border-style":"solid",
                    "border-radius": "3px",
                    "margin":"1"
                    }
                    )

def get_metric_page_graph():

    


     
     
    layout =   html.Div([
    
        dcc.Store(id='legend_data',  storage_type='local'),
        # dcc.Dropdown( id = 'id_checklist_inv',multi =True,  options = [], value=[], persistence=True,persistence_type='session' ,style={'display': 'block'}) , 

        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
         ),

        
        html.Div( 
        
            html.Div( 
                [
                    get_selectors() 
            ])

            ,

           
            # style = {
            #     # "background-color":"rgb(198,216,233)",
            #     "border-width": "thin",
            #     "border-color":"silver",
            #     "border-style":"solid",
            #     "border-radius": "10px",
            #     "padding":"10px",
            #      "border-radius": "3px",
            #     }
            ),
    
            get_graph()
           
        ])

    return layout
 

@app.callback(
    Output('select_metric_dd', 'options'),
    Output('select_metric_dd', 'value'), 
    Input('select_group', 'value') 
)
def update_metrics(value):
     
    if not value:
        logger.info("no value for update")
        return [], None
        # return children
        return dcc.Dropdown( [],  id='select_metrics',persistence=True, persistence_type='session')
        
     
    metrics = []
    for m in redisUtil.get_metric_info()[value]:
        if m == "meta": 
            continue
        if redisUtil.get_metric_info()[value][m]["type"] in ["numeric"]:
            metrics.append(m)
    
    
    return metrics,None
    # return  dcc.Dropdown( metrics,  id='select_metrics',persistence=True, persistence_type='session') 
  




@app.callback(
    Output('legend_data', 'data'),
    Input('select_metric_dd', 'value') ,
    Input('reset_graph', "n_clicks"),
    State('select_group', 'value') , 
    State('legend_data', 'data')  
)
def update_options(metric,clk, group, options):

    # if not options:
    #     options  = ['group_fairness,average_odds_difference'] 
    ctx = dash.callback_context
     
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'reset_graph.n_clicks':
        return  []
        
    if metric is None or group is None:
        return options
    
    item = group+"," + metric
    if item not in options:
        options.append(item)
    
    return options
     



@app.callback(
    Output('metric_graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('legend_data', 'data'),
    State('metric_graph', 'figure')
)
def update_graph( n,options,old):

    
    # print(n,options,old)
    ctx = dash.callback_context
      
    
    if 'prop_id' in ctx.triggered[0] and ctx.triggered[0]['prop_id'] == 'interval-component.n_intervals':
        
        if redisUtil.has_update("metric_graph", reset = True):
            logger.info("new data")
            redisUtil._subscribers["metric_graph"] = False
            
        else:
            # logger.info("ignore timer")
            return old

    fig = go.Figure()
    
    print('options === ', options)

    for item in options:

        k,v = item.split(',')
        print(k,v)
        print('---------------------------')
        tags, sc_data = get_trc_data(k, v) 
        fig.add_traces( go.Scatter(**sc_data))

    if not options:
        return fig
 
    fig.update_traces(textposition="top center")
    fig.update_layout(
        
        xaxis = dict(
            tickmode = 'array',
            tickvals =  sc_data["x"],
            ticktext = tags
            
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
    
     
    return fig
