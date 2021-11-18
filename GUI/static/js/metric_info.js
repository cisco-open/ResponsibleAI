'use strict';
var whitelist = []; // contains all metrics whose category is checked
var metrics; // global variable contains all metrics on screen
var graphs = {}; // list of all graphs
var matrices = {};
var metric_data;
var metric_name;
var metric_has_range;
var metric_range;
var metric_display_name;
var metric_type;

var tags = {}


// Queries Data
function loadData(metric_name, metric_range_, metric_display_name_, metric_type_, metric_has_range_) {
    metric_name = metric_name.replace('&gt;', '>')
    metric_range = metric_range_;
    metric_display_name = metric_display_name_;
    metric_type = metric_type_;
    metric_has_range = metric_has_range_;

    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            callAllFunctions(metric_name, text);
        });
}

// Use collected data to create metrics, boxes and white list metrics by category
function callAllFunctions(metric_name, data) {
    createMetrics(metric_name, data);
}

// Create graphs
function createMetrics(metric_name, data) {
    if(metric_type == 'numeric'){
        addChart(metric_name, data, "");
    }
    else if(metric_type == "vector"){
        addChart(metric_name, data, "-single")
    }
    else if(metric_type == "matrix"){
        var res = stringToMatrix(data, metric_name);
        addTable(metric_name, res);
    }
    else if(metric_type == "boolean"){
        addBoolChart(metric_name, data);
    }
}


function stringToMatrix(data, name){
    var result = []
    if (data.length >= 1)
        result = data[data.length-1][name];
    return result
}


function addChart(metric_name, data, name_extension){
    var body = document.getElementById('_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", 'MetricPage chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");
    var writing = document.createElement('p');
    writing.innerHTML = metric_display_name;
    writing.setAttribute("class", "chartHeader");
    var writing2 = document.createElement('p');
    if (typeof(data[data.length-1][metric_name + name_extension]) == 'number')
        writing2.innerHTML = data[data.length -1][ metric_name + name_extension].toFixed(3);
    else
        writing2.innerHTML = "Null"
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");
    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    var chart = document.createElement('div');
    chart.id = metric_name;
    chart.setAttribute("class", "morris-chart chartScalerSmall");
    newDiv.appendChild(chart);
    body.appendChild(newDiv);

    var myValues  = {
        element: metric_name,
        data: createData(data, metric_name + name_extension),
        xkey: 'year',
        ykey: metric_name,
        hideHover: true,
        smooth: false,
        lineColors: ['#000000'],
        pointFillColors: ['#000000'],
        ykeys: ['value'],
        labels: ['Value']
    }


    if(metric_has_range == 'True'){
        var range = JSON.parse(metric_range)
        if(range[0] != null){
            myValues['ymin'] = range[0]
        }
        if(range[1] != null){
            myValues['ymax'] = range[1]
        }
    }
    var morrisLine = new Morris.Line(myValues)
    graphs[metric_name] = morrisLine;
}


function addBoolChart(metric_name, data){
    var body = document.getElementById('_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", 'MetricPage chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");
    var writing = document.createElement('p');
    writing.innerHTML = metric_display_name;
    writing.setAttribute("class", "chartHeader");
    var writing2 = document.createElement('p');
    if(data[data.length-1][metric_name + name_extension] == null)
        writing2.innerHTML = "Null"
    else
        writing2.innerHTML = data[data.length -1][ metric_name + name_extension];
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");

    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    var chart = document.createElement('div');
    chart.id = metric_name;
    chart.setAttribute("class", "morris-chart chartScalerSmall");
    newDiv.appendChild(chart);


    body.appendChild(newDiv);
}


function addTable(metric_name, data_array){
    var body = document.getElementById('_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", 'MetricPage chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");
    var writing = document.createElement('p');
    writing.innerHTML = metric_display_name
    writing.setAttribute("class", "chartHeader");
    newDiv.appendChild(writing);

    var chart = document.createElement('div');
    chart.setAttribute("class", "overflow_table")
    chart.id = metric_name;
    newDiv.appendChild(chart);
    body.appendChild(newDiv);

    var table = generateTableFromArray(data_array)
    chart.appendChild(table);
    matrices[metric_name] = chart;
}

function generateTableFromArray(data_array){
    var table = document.createElement('table');
    table.setAttribute('class', 'displayMatrix')
    if(data_array == null)
        return table
    var tableBody = document.createElement('tbody');
    tableBody.setAttribute('class', 'displayMatrix');
    for(var r = 0; r < data_array.length; r++){
        var row = document.createElement('tr');
        row.setAttribute('class', 'displayMatrix')
        for(var c = 0; c < data_array[r].length; c++){
            var col = document.createElement('td');
            col.setAttribute('class', 'displayMatrix')
            col.appendChild(document.createTextNode(data_array[r][c]));
            row.appendChild(col);
        }
        tableBody.appendChild(row);
    }
    table.appendChild(tableBody);
    return table
}

// Used to create the data for the morris chart
function createData(data, key) {
    var ret = [];
    for (var i = 0; i < data.length; i++) {
        if(data[i][key] != null && !isNaN(data[i][key]) && isFinite(data[i][key])){
            ret.push({
                year: data[i]["date"],
                value: data[i][key]
            });
        }
    }
    return ret;
}


// Reload the metrics once the times to query for are changed
function redoMetrics() {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            redoMetrics2(text)
        });
}

// Fill graphs with new data
function redoMetrics2(data) {
    for (var type in graphs) {
        var ext = "";
        if(metric_type == "vector")
            ext = "-single"
        var new_data = createData(data, type + ext);
        graphs[type].setData(new_data);
        var writing = document.getElementById(type + "LastValue");

        if(new_data.length >= 1)
            writing.innerHTML = new_data[new_data.length - 1]["value"].toFixed(3);
        else
            writing.innerHTML = ""
    }
    for (var type in matrices){
        var chart = document.getElementById(type + "_chart")
        var hiddenText = chart.id.substring(0, chart.id.indexOf("_chart"))
        var internalDiv = chart.getElementsByTagName("div")[0]
        var table = internalDiv.getElementsByTagName("table")[0]
        table.remove()

        var res = stringToMatrix(data, hiddenText)
        var table = generateTableFromArray(res)
        internalDiv.appendChild(table);
    }
}
