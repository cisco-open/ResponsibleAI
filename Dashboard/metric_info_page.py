import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
from server import app, redisUtil

 
from dash import dash_table
import pandas as pd

tbl_styling = { 
    'style_data':{
        'color': 'black',
        'backgroundColor': 'white',
        'textAlign': 'left',
        "whiteSpace": "pre-line"
    },
    'style_data_conditional':[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
    ],
    'style_header':{
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold',
        'textAlign': 'center'
    }}


g_config = []
def get_tbl():

    d = {"Metric Group Name":[],"tags":[],"dependency list":[], "complexity":[]}
    for g,v in redisUtil.info["metric_info"].items():
        d["Metric Group Name"].append(g)
        d["tags"].append(", ".join(v["meta"]["tags"]))
        d["complexity"].append(v["meta"]["complexity_class"])
        d["dependency list"].append("")
    df = pd.DataFrame(data = d)
    
    return dash_table.DataTable(
        id = "groups",
        data = df.to_dict('records'), 
        columns = [{"name": i, "id": i} for i in df.columns],
        **tbl_styling
     )

def get_tbl2(config):
    table_header = [
    html.Thead(html.Tr([html.Th("Metric Group Name"), html.Th("Tags"), html.Th("dependency list"), html.Th("complexity")]))
    ]

    rows=[]
    for g,v in config["redis"].info["metric_info"].items():
        rows.append( html.Tr( [
                                html.Td(g),
                                html.Td(  ", ".join(v["meta"]["tags"])),
                                html.Td(v["meta"]["dependency_list"]),
                                html.Td(v["meta"]["complexity_class"])
                            ]) )

    table_body = [html.Tbody(rows)]
    table = dbc.Table(table_header + table_body, bordered=True,
    dark=False,
    hover=True,
    responsive=True,
    striped=True,)
    return table

 
 
def get_metrics(group):
     
    d = {"Metric Name":[],"tags":[],"type":[], "range":[],"explanation":[]}
    
    for m,v in redisUtil.info["metric_info"][group].items():
        if m=="meta":
            continue
        d["Metric Name"].append(v['display_name'])
        if "tags" in v:
            d["tags"].append(", ".join(v["tags"]))
        else:
            d["tags"].append("")
        d["type"].append(v["type"])
        vr = v["range"]
        d["range"].append( f"{vr}" )
        d["explanation"].append(v["explanation"])
    df = pd.DataFrame(data = d)
    
    return dash_table.DataTable(df.to_dict('records'), 
    [{"name": i, "id": i} for i in df.columns],
    id = "metrics" ,
    **tbl_styling)

def get_metric_info_page():
    
    return html.Div( [
        html.P("metric groups"),
        get_tbl(),
        html.P("metrics info"),
        html.Div( id = "metrics_div")
    ])

@app.callback(
    Output('metrics_div', 'children'),
    Input('groups', 'active_cell'),
    State('groups', 'data') 
)
def update(active_cell,data):
    
    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        group_name = data[row]["Metric Group Name"]

        # print("group name = ", group_name)
        return [get_metrics(group_name)]




