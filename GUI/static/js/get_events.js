'use strict';

var metric_file = "output/metric_list.json";
var explanation_file = "/Flask-Admin-Dashboard/static/output/metric_info.json";


var whitelist = []; // contains all metrics whose category is checked
var blacklist = []; // contains which metrics were X'd out
var metrics; // global variable contains all metrics on screen
var graphs = {}; // list of all graphs
var matrices = {};
var metric_info;
var tags = {}

// Queries Data
function load_data() {
    var date1 = '0000-00-00';
    var today = new Date()
    var date2 = today.getFullYear() + "-" + String(today.getMonth() + 1).padStart(2, '0') + "-" + String(today.getDate()).padStart(2, '0');
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            fillTable(text);
        });
}


// Populate events
function fillTable(measurements) {
    if(measurements == null)
        return
    var table = document.getElementById('event_list');

    for(var i = measurements.length-1; i>= 0; i--){
        var row = document.createElement('tr');
        row.setAttribute('class', 'displayMatrix')

        var col1 = document.createElement('td')
        col1.appendChild(document.createTextNode(measurements[i]['metadata > date']))
        row.appendChild(col1)

        var col2 = document.createElement('td')
        col2.appendChild(document.createTextNode('Measurement Added'))
        row.appendChild(col2)

        var col3 = document.createElement('td')
        col3.appendChild(document.createTextNode(measurements[i]['metadata > description']))
        row.appendChild(col3)

        table.appendChild(row);
    }
}


function stringToMatrix(data, name){
    var result = []
    if (data.length >= 1)
        result = data[data.length-1][name];
    return result
}


function addTable(metric_name, explanations, data_array, category){
    addTags(metric_name)
    var body = document.getElementById('metric_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", category.toLowerCase() + 'Metric col-sm-6 chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");
    var writing = document.createElement('p');
    writing.innerHTML = metric_info[metric_name]["display_name"]
    writing.setAttribute("class", "chartHeader");
    var img = document.createElement('img');
    img.setAttribute("title", explanations[metric_name]["explanation"]);
    img.setAttribute("src", "/static/img/questionMark.png");
    img.setAttribute("alt", "Learn more about " + metric_name);
    img.setAttribute("class", "learnMore");
    newDiv.appendChild(img);
    newDiv.appendChild(writing);

    var removeBtn = document.createElement("button");
    removeBtn.innerHTML  = "X";
    removeBtn.setAttribute("class", "removeChart");
    removeBtn.setAttribute("style", "display:none");
    removeBtn.setAttribute("onclick", "blackList('" + metric_name + "')");
    newDiv.appendChild(removeBtn);

    var link = document.createElement('a')
    link.setAttribute('href', '/learnMore/'+metric_name)
    link.setAttribute('class', 'learnMoreLink')
    var logo = document.createElement('i')
    logo.setAttribute('class', 'fa fa-external-link fa-lg')
    link.appendChild(logo)
    newDiv.appendChild(link)

    var chart = document.createElement('div');
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
            if(Number.isInteger(data_array[r][c]))
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
