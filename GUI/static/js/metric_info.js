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
var model_info;
var bool_charts = {}
var dict_charts = {}
var tags = {}
var use_date = true;
var page_ready = false


$(document).ready(function() {
        setInterval("check_data()", 1000); // call every 10 seconds
});

function check_data() {
    if(page_ready){
       fetch('/updateMetrics').then(function (response) {
            return response.json();
        }).then(function(result){
            if (result){
                redoMetrics();
            }
        });
    }
}



// Queries Data
function loadData(metric_name, metric_range_, metric_display_name_, metric_type_, metric_has_range_) {
    metric_name = metric_name.replace('&gt;', '>')
    metric_range = metric_range_.replace("None", "null");
    metric_display_name = metric_display_name_;
    metric_type = metric_type_;
    metric_has_range = metric_has_range_;
    page_ready = false
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            load_model_info(metric_name, text);
        });
}

// Loads explanations.
function load_model_info(metric_name, data) {
    fetch('/getModelInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        model_info = text;
        callAllFunctions(metric_name, data);
    });
}


// Use collected data to create metrics, boxes and white list metrics by category
function callAllFunctions(metric_name, data) {
    createMetrics(metric_name, data);
    page_ready = true
}

// Create graphs
function createMetrics(metric_name, data) {
    if(metric_type == 'numeric'){
        addChart(metric_name, data, "");
    }
    else if(metric_type == "vector"){
        if(metric_name.indexOf("_avg") >= 0)
            addChart(metric_name, data, "-single")
        else {
                res = stringToMatrix(data, metric_name)
                if (!Array.isArray(res[0]))
                    res = [res]
                addTable(metric_name, res)
        }
    }
    else if(metric_type == "matrix"){
        var res = stringToMatrix(data, metric_name);
        addTable(metric_name, res);
    }
    else if(metric_type == "boolean"){
        addBoolChart(metric_name, data);
    }
    else if(metric_type == "vector-dict"){
        addVectorDict(metric_name, data);
    }
}



function addVectorDict(metric_name, data){
    var curData = data[data.length -1][metric_name]
    var features = model_info['features']
    var result = {}
    for(var i = 0; i<curData.length; i++){
        if(curData[i] != null){
            var table = dict_to_table(curData[i])
            addTable(metric_name, table, features[i], String(i));
        }
    }
}


function dict_to_table(dict){
    var result = [[], []]
    for(var key in dict){
        result[0].push(key)
        result[1].push(dict[key])
    }
    return result
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

    var result = createData(data, metric_name + name_extension)
    var chart_data = result[0]
    var chart_descriptions = result[1]
    var myValues  = {
        element: metric_name,
        data: chart_data,
        xkey: 'year',
        descriptions: chart_descriptions,
        ykey: metric_name,
        hideHover: true,
        smooth: false,
        lineColors: ['#000000'],
        pointFillColors: ['#000000'],
        ykeys: ['value'],
        labels: ['Value'],
        hoverCallback: function (index, options, content, row) {
                var description = options.descriptions[index];
                return content + "\nDescription: " + description;}
    }

    if(metric_has_range == 'True'){
        var range = JSON.parse(metric_range)
        if(range[0] != null){
            myValues['ymin'] = range[0]
        }
        if(range[1] != null){
            myValues['ymax'] = range[1]
        }
        myValues['yLabelFormat'] = function(y){return y.toFixed(2);}
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
    if(data[data.length-1][metric_name] == null)
        writing2.innerHTML = "Null"
    else
        writing2.innerHTML = data[data.length -1][ metric_name];
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");

    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    var chart = document.createElement('div');
    chart.setAttribute("class", "overflow_table")
    chart.id = metric_name;
    chart.setAttribute("class", "morris-chart chartScalerSmall");
    newDiv.appendChild(chart);
    body.appendChild(newDiv);


    var result = createBoolData(data, metric_name)
    var chart_data = result[0]
    var chart_descriptions = result[1]
    var myValues  = {
        element: metric_name,
        data: chart_data,
        xkey: 'year',
        descriptions: chart_descriptions,
        ykey: metric_name,
        hideHover: true,
        smooth: false,
        lineColors: ['#000000'],
        pointFillColors: ['#000000'],
        ykeys: ['value'],
        labels: ['Value'],
        hoverCallback: function (index, options, content, row) {
                var description = options.descriptions[index];
                return content + "\nDescription: " + description;}
    }
    myValues['parseTime'] = true
    var morrisLine = new Morris.Line(myValues)

    bool_charts[metric_name] = morrisLine;
    body.appendChild(newDiv);
}


function addTable(metric_name, data_array, optionalName="", optionalNumber=""){
    var body = document.getElementById('_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", 'MetricPage chart-container main-panel');
    if(optionalNumber!="")
        optionalNumber = "|"+optionalNumber;
    newDiv.setAttribute("id", metric_name + "_chart"+optionalNumber);
    var writing = document.createElement('p');
    writing.innerHTML = metric_display_name
    if(optionalName != "")
        writing.innerHTML += " - " + optionalName
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

function generateTableFromArray(data_array, is_float=false){
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
            if(typeof data_array[r][c] == 'string' || data_array[r][c] instanceof String || Number.isInteger(data_array[r][c]))
                col.appendChild(document.createTextNode(data_array[r][c]));
            else
                col.appendChild(document.createTextNode(data_array[r][c].toFixed(2)));
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
    var descriptions = []
    for (var i = 0; i < data.length; i++) {
        if(data[i][key] != null && !isNaN(data[i][key]) && isFinite(data[i][key])){
            if(use_date){
                ret.push({
                    year: data[i]["metadata > date"],
                    value: data[i][key]
                });
            }
            else{
                ret.push({
                    year: data[i]["metadata > description"],
                    value: data[i][key]
                });
            }
            descriptions.push(data[i]["metadata > description"])
        }
    }
    return [ret, descriptions];
}

// Used to create the data for the morris chart
function createBoolData(data, key) {
    var ret = [];
    var descriptions = []
    for (var i = 0; i < data.length; i++) {
        if(data[i][key] != null && !isNaN(data[i][key]) && isFinite(data[i][key])){
            var value = 0
                if(data[i][key])
                    value = 1
            if(use_date){
                ret.push({
                    year: data[i]["metadata > date"],
                    value: value
                });
            }
            else{
                ret.push({
                    year: data[i]["metadata > description"],
                    value: value
                });
            }
            descriptions.push(data[i]["metadata > description"])
        }
    }
    return [ret, descriptions];
}


// Reload the metrics once the times to query for are changed
function redoMetrics() {
    page_ready = false
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            redoMetrics2(text)
            page_ready = true
        });
}

function redoMetrics2(data) {
    for (var type in graphs) {
        var ext = "";
        if(metric_type == "vector")
            ext = "-single"
        var result = createData(data, type + ext);
        var new_data = result[0]
        var newExplanations = result[1]
        graphs[type]['options'].parseTime = use_date
        graphs[type].setData(new_data);
        graphs[type].options.descriptions = newExplanations

        var writing = document.getElementById(type + "LastValue");
        if(new_data.length >= 1)
            writing.innerHTML = new_data[new_data.length - 1]["value"].toFixed(3);
        else
            writing.innerHTML = "Null"
    }
    for (var type in bool_charts){
        var writing2 = document.getElementById(type + "LastValue");
        if(data.length == 0 || data[data.length-1][type] == null){
            writing2.innerHTML = "Null"
        }
        else{
            writing2.innerHTML = data[data.length -1][type];
        }
        var result = createBoolData(data, type);
        var new_data = result[0]
        var newExplanations = result[1]
        bool_charts[type].options.parseTime = use_date
        bool_charts[type].setData(new_data);
        bool_charts[type].options.descriptions = newExplanations
    }
    for (var type in matrices){
        if(metric_type == 'matrix' || metric_type == 'vector'){
            var chart = document.getElementById(type + "_chart")
            var hiddenText = chart.id.substring(0, chart.id.indexOf("_chart"))
            var internalDiv = chart.getElementsByTagName("div")[0]
            var table = internalDiv.getElementsByTagName("table")[0]
            table.remove()
            var res = stringToMatrix(data, hiddenText)
            if (!Array.isArray(res[0]))
                        res = [res]
            var table = generateTableFromArray(res)
            internalDiv.appendChild(table);
        }
        if(metric_type == 'vector-dict'){
            var curData = []
            if (data.length >= 1){
                curData = data[data.length -1][type]
            }
            var features = model_info['features']
            var result = {}
            for(var i = 0; i<features.length; i++){
                var chart = document.getElementById(type + "_chart" + "|" + String(i))
                if(chart != null){
                    var hiddenText = chart.id.substring(0, chart.id.indexOf("_chart"))
                    var internalDiv = chart.getElementsByTagName("div")[0]
                    var table = internalDiv.getElementsByTagName("table")[0]
                    table.remove()
                    var table_data = dict_to_table(curData[i]);
                    table_data = generateTableFromArray(table_data)
                    internalDiv.appendChild(table_data);
                }
                else if(curData.length != 0 && curData[type] != null){  // Add elements which may have previously been null
                    addVectorDict(type, metric_info, data, metric_info[type]["category"], "");
                }
            }
        }
    }
    for (var type in bool_charts){
        var writing2 = document.getElementById(type + "LastValue");
        if(data[data.length-1][type] == null){
            writing2.innerHTML = "Null"
        }
        else{
            writing2.innerHTML = data[data.length -1][type];
        }
    }
}


function date_slider(){
    var slider = document.getElementById('slider_input')
    use_date = !slider.checked;
    redoMetrics()
}