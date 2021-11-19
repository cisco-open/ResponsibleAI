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

// loads metrics. Contains lists of metrics, and metric tags.
function loadMetrics(category) {
    fetch('/getMetricList').then(function (response) {
        return response.json();
    }).then(function(text){
        metrics = text
        loadExplanations(text, category);
    });
}

// Loads explanations.
function loadExplanations(metrics, category) {
    fetch('/getMetricInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        metric_info = text;
        load_data(metrics, text, category);
    });
}

// Queries Data
function load_data(metrics, data, category) {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            callAllFunctions(metrics, data, text, category);
        });
}

// Use collected data to create metrics, boxes and white list metrics by category
function callAllFunctions(metrics, data, df_json, category) {
    createMetrics(metrics, data, df_json, category);
    createBoxes(metrics, category);
    createWhiteList(metrics, category);
}


// Create graphs
function createMetrics(metrics, explanations, data, category) {
    var list = metrics[category.toLowerCase()];
    for (var i = 0; i < list.length; i++) {
        if(metric_info[list[i]]["type"] == "numeric"){
            addChart(list[i], explanations, data, category, "");
        }
        else if(metric_info[list[i]]["type"] == "vector"){
            if(list[i].indexOf("_avg") >= 0)
                addChart(list[i], explanations, data, category, "-single")
            else if (! (list[i]+"_avg" in metric_info)){
                res = stringToMatrix(data, list[i])
                if (!Array.isArray(res[0]))
                    res = [res]
                addTable(list[i], explanations, res, category)
            }
        }
        else if(metric_info[list[i]]["type"] == "matrix"){
            var res = stringToMatrix(data, list[i])
            addTable(list[i], explanations, res, category)
        }
        else if(metric_info[list[i]]["type"] == "boolean"){
            addBoolChart(list[i], explanations, data, category, "");
        }
    }
}


function stringToMatrix(data, name){
    var result = []
    if (data.length >= 1)
        result = data[data.length-1][name];
    return result
}

function addTags(metric_name){
    for (var i = 0; i < metric_info[metric_name]["tags"].length; i++){
        if (tags[metric_info[metric_name]["tags"][i]] == null)
            tags[metric_info[metric_name]["tags"][i]] = [];
        tags[metric_info[metric_name]["tags"][i]].push(metric_name);
    }
}


function addChart(metric_name, explanations, data, category, name_extension){
    addTags(metric_name)
    // console.log(metric_name)
    var body = document.getElementById('metric_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", category.toLowerCase() + 'Metric col-sm-6 chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");
    var writing = document.createElement('p');
    writing.innerHTML = metric_info[metric_name]["display_name"];
    writing.setAttribute("class", "chartHeader");
    var writing2 = document.createElement('p');
    if (typeof(data[data.length-1][metric_name + name_extension]) == 'number')
        writing2.innerHTML = data[data.length -1][ metric_name + name_extension].toFixed(3);
    else
        writing2.innerHTML = "Null"
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");
    var img = document.createElement('img');
    img.setAttribute("title", explanations[metric_name]["explanation"]);
    img.setAttribute("src", "/static/img/questionMark.png");
    img.setAttribute("alt", "Learn more about " + metric_name);
    img.setAttribute("class", "learnMore");
    var link = document.createElement('a')
    link.setAttribute('href', '/learnMore/'+metric_name)
    link.setAttribute('class', 'learnMoreLink')
    var logo = document.createElement('i')
    logo.setAttribute('class', 'fa fa-external-link fa-lg')
    link.appendChild(logo)
    newDiv.appendChild(link)

    newDiv.appendChild(img);
    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    var removeBtn = document.createElement("button");
    removeBtn.innerHTML  = "X";
    removeBtn.setAttribute("class", "removeChart");
    removeBtn.setAttribute("style", "display:none");
    removeBtn.setAttribute("onclick", "blackList('" + metric_name + "')");
    newDiv.appendChild(removeBtn);

    var chart = document.createElement('div');
    chart.setAttribute("class", "overflow_table")
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
    if(metric_info[metric_name]["has_range"]){
        if(metric_info[metric_name]["range"][0] != null){
            myValues['ymin'] = Number(metric_info[metric_name]["range"][0])
        }
        if(metric_info[metric_name]["range"][0] != null){
            myValues['ymax'] = Number(metric_info[metric_name]["range"][1])
        }
    }
    var morrisLine = new Morris.Line(myValues)
    graphs[metric_name] = morrisLine;
}


function addBoolChart(metric_name, explanations, data, category, name_extension){
    addTags(metric_name)
    // console.log("DATA: " + JSON.stringify(data))
    var body = document.getElementById('metric_row');
    var newDiv = document.createElement('div');
    newDiv.setAttribute("class", category.toLowerCase() + 'Metric col-sm-6 chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");
    var writing = document.createElement('p');
    writing.innerHTML = metric_info[metric_name]["display_name"];
    writing.setAttribute("class", "chartHeader");
    var writing2 = document.createElement('p');
    if(data[data.length-1][metric_name + name_extension] == null)
        writing2.innerHTML = "Null"
    else
        writing2.innerHTML = data[data.length -1][ metric_name + name_extension];
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");
    var img = document.createElement('img');
    img.setAttribute("title", explanations[metric_name]["explanation"]);
    img.setAttribute("src", "/static/img/questionMark.png");
    img.setAttribute("alt", "Learn more about " + metric_name);
    img.setAttribute("class", "learnMore");
    var link = document.createElement('a')
    link.setAttribute('href', '/learnMore/'+metric_name)
    link.setAttribute('class', 'learnMoreLink')
    var logo = document.createElement('i')
    logo.setAttribute('class', 'fa fa-external-link fa-lg')
    link.appendChild(logo)
    newDiv.appendChild(link)

    newDiv.appendChild(img);
    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    var removeBtn = document.createElement("button");
    removeBtn.innerHTML  = "X";
    removeBtn.setAttribute("class", "removeChart");
    removeBtn.setAttribute("style", "display:none");
    removeBtn.setAttribute("onclick", "blackList('" + metric_name + "')");
    newDiv.appendChild(removeBtn);

    var chart = document.createElement('div');
    chart.id = metric_name;
    chart.setAttribute("class", "morris-chart chartScalerSmall");
    newDiv.appendChild(chart);


    body.appendChild(newDiv);
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

// Create the category searching of metrics
function createBoxes(metrics, category) {
    var body = document.getElementById('selector');
    var list = tags
    var topBox = document.createElement("input");
    topBox.setAttribute("type", "checkbox");
    topBox.setAttribute("id", category + "_mainBox");
    topBox.setAttribute("value", true);
    topBox.setAttribute("name", category + "_mainBox");
    topBox.setAttribute("class", "selectorBox");
    topBox.setAttribute("class", "parentBox");
    topBox.setAttribute("checked", true);
    topBox.setAttribute("onclick", "checkChlidren(this, '" + category.toString().toLowerCase() + "')");
    var topLabel = document.createElement("label");
    topLabel.setAttribute("for", category + "_mainBox");
    topLabel.innerHTML = category;
    var topBr = document.createElement("br");
    body.appendChild(topBox);
    body.appendChild(topLabel);
    body.appendChild(topBr);
    for (var i in tags) {
        if (i == category.toLowerCase())
            continue;

        var newBox = document.createElement("input");
        newBox.setAttribute("type", "checkbox");
        newBox.setAttribute("id", i);
        newBox.setAttribute("value", true);
        newBox.setAttribute("name", i);
        newBox.setAttribute("class", "selectorBox");
        newBox.setAttribute("class", category.toString().toLowerCase() + "Box");
        newBox.setAttribute("checked", true);
        newBox.setAttribute("onclick", "checkParent(this, '" + category + "')");
        var label = document.createElement("label");
        label.setAttribute("for", i + category.toString().toLowerCase());
        label.innerHTML = i;
        var br = document.createElement("br")
        body.appendChild(newBox);
        body.appendChild(label);
        body.appendChild(br);
    }
    var button = document.createElement("button");
    button.setAttribute("class", "selectorButton");
    button.innerHTML = "Done";
    button.setAttribute("onclick", "doneEdit('" + category + "');")
    body.appendChild(button);
}

// white list metrics depending on what is checked in the categories
function createWhiteList(metrics, category) {
    for (var i in tags)
        for (var j = 0; j<tags[i].length; j++){
            whitelist.push(tags[i][j]);
        }
}

// Used to display the edit view menu
function displayMenu(classtype) {
    document.getElementById("selector").style.display = "";
    var list = document.getElementsByClassName("removeChart");
    for (var i = 0; i < list.length; i++)
        list[i].style.display = "";
}

// Removes a metric when its X is pressed
function blackList(name) {
    blacklist.push(name);
    document.getElementById(name + "_chart").style.display = "none";
}

// Hides the selector menu
function doneEdit(classtype) {
    document.getElementById("selector").style.display = "none";
    var list = document.getElementsByClassName("removeChart");
    for (var i = 0; i < list.length; i++)
        list[i].style.display = "none";
}

// Generates the white list (what metrics should be shown, based on category) by looking at the checked boxes
function generateWhiteList(classtype) {
    var boxes = document.getElementsByClassName(classtype.toString().toLowerCase() + "Box");
    whitelist = [];

    if(boxes.length == 0){
        var box = document.getElementById(classtype.toString().toLowerCase() + "_mainBox");
        if (box.checked){
            var id = box.id.toString();
            id = id.substring(0, id.indexOf("_"))
            var list = tags[id];
            for (var j = 0; j < list.length; j++) {
                whitelist.push(list[j]);
            }
        }
    }
    else{
        for (var i = 0; i < boxes.length; i++) {
            if (boxes[i].checked) {
                var id = boxes[i].id.toString();
                var list = tags[id];
                for (var j = 0; j < list.length; j++) {
                    whitelist.push(list[j]);
                }
            }
        }
    }
    displayWhiteList(classtype)
}

// Display the white listed metrics
function displayWhiteList(classtype) {
    var row = document.getElementById("metric_row");
    var divs = row.getElementsByClassName(classtype.toString().toLowerCase() + "Metric");
    for (var i = 0; i < divs.length; i++) {
        var div = divs[i].getElementsByTagName("div")[0];
        if (whitelist.includes(div.id) && !blacklist.includes(div.id)) {
            divs[i].style.display = "";
        }
        else {
            divs[i].style.display = "none";
        }
    }
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
        if(metric_info[type]["type"] == "vector")
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


// Checks child status of boxes once parent boxes are checked in metric selector
function checkChlidren(check, type) {
    if ($(check).is(':checked')) {
        blacklist = [];
        var all = $("." + type + "Box").map(function () {
            this.checked = true;
        }).get();
    }
    else {
        var all = $("." + type + "Box").map(function () {
            this.checked = false;
        }).get();
    }
    generateWhiteList(type);
}

// Checks parent status in checkboxes, once all child checks are unchecked, or one is checked.
function checkParent(check, type) {
    var count = 0;
    if ($(check).is(':checked')) {
        blacklist = [];
        var all = $("#" + type + "_mainBox").map(function () {
            this.checked = true;
        }).get();
    }
    else {
        var swap = false;
        var items = document.getElementsByClassName(type.toString().toLowerCase() + "Box");
        for (var i = 0; i < items.length; i++) {
            swap = items[i].checked;
            if (swap)
                break
        }
        if (!swap)
            document.getElementById(type + "_mainBox").checked = false;
    }
    generateWhiteList(type);
}


// Searches available metrics, changes display based on search results.
function search(category) {
    var input, filter, row, divs, i, text;
    input = document.getElementById("myInput");
    filter = input.value.toLowerCase();
    row = document.getElementById("metric_row");
    divs = row.getElementsByTagName("div");

    var mustInclude = [];
    for (var i in tags) {
        if (i.toLowerCase().indexOf(filter) > -1) {
            for(var j = 0; j<tags[i].length; j++) {
                mustInclude.push(tags[i][j]);
            }
        }
    }
    for (var i = 0; i < divs.length; i++) {
        var p = divs[i].getElementsByClassName("chartHeader")[0];
        var hiddenDiv = divs[i].getElementsByTagName("div")[0];
        if (p && hiddenDiv) {
            var text = p.innerText;
            var hiddenText = hiddenDiv.id;
            if ((text.toLowerCase().indexOf(filter) > -1 || mustInclude.includes(hiddenText) && whitelist.includes(hiddenText)) && !blacklist.includes(hiddenText)) {
                divs[i].style.display = "";
            } else {
                divs[i].style.display = "none";
            }
        }
    }
}


