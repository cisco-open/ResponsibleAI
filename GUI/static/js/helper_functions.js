'use strict';


// Create all graphs for a category of metrics
// Accepts metrics, which contains all metrics, metadata, metric explanations, the data, the desired category,
// Consider removing remaining on all remaining ones.
function createMetrics(metrics, metric_info, explanations, data, category, add_tag, tagOwner, tag_list, data_types) {
    var list = metrics[category.toLowerCase()];
    if (list==null)
        return
    for (var i = 0; i < list.length; i++) {
        createMetric(list[i], metric_info[list[i]]["type"], metric_info, explanations, data, category, add_tag, metric_info[list[i]]["tags"], tagOwner, tag_list, data_types)
    }
}


// Creates an individual metric. This part looks at a metrics metadata and determines which function should be used to display it.
function createMetric(metric_name, metric_type, metric_info, explanations, data, category, add_tag=true, tags=[], tagOwner, tag_list, data_types){
    if(metric_type == "numeric"){ // Numeric metrics are displayed as charts.
        addChart(metric_name, metric_info[metric_name]["display_name"], explanations[metric_name]["explanation"],
                data, category, "", metric_info[metric_name]["has_range"], metric_info[metric_name]["range"], add_tag, tags, tagOwner, tag_list, data_types);
    }
    else if(metric_type == "vector"){ // Vector metrics that also contain a _avg metric are displayed as charts.
        if(metric_name.indexOf("_avg") >= 0)
            addChart(metric_name, metric_info[metric_name]["display_name"], explanations[metric_name]["explanation"],
                data, category, "-single", metric_info[metric_name]["has_range"], metric_info[metric_name]["range"], add_tag, tags, tagOwner, tag_list, data_types);
        else if (! (metric_name+"_avg" in metric_info)){ // Vector metrics that don't have an _avg metric are displayed as tables
            res = stringToMatrix(data, metric_name)
            if (res != null && !Array.isArray(res[0]))
                res = [res]
            addTable(metric_name, metric_info[metric_name]["display_name"], explanations[metric_name]["explanation"], res, category, "", "", add_tag, tags, tagOwner, tag_list, data_types)
        }
    }
    else if(metric_type == "matrix"){ // Matrices are displayed as tables.
        var res = stringToMatrix(data, metric_name)
        addTable(metric_name, metric_info[metric_name]["display_name"], explanations[metric_name]["explanation"], res, category, "", "", add_tag, tags, tagOwner, tag_list, data_types)
    }
    else if(metric_type == "boolean"){ // Booleans are displayed as charts, using BoolChart because we need to handle data differently.
        addBoolChart(metric_name, metric_info[metric_name]["display_name"], explanations[metric_name]["explanation"],
                data, category, "", metric_info[metric_name]["has_range"], metric_info[metric_name]["range"], add_tag, tags, tagOwner, tag_list, data_types);
    }
    else if(metric_type == "vector-dict"){ // Vector dicts are vectors of dictionaries. It is currently assumed that each index associates with a feature of the dataset.
        addVectorDict(metric_name, metric_info[metric_name]["display_name"], explanations[metric_name]["explanation"],
            model_info['features'], data, category, add_tag, tags, tagOwner, tag_list, data_types);
    }
}


// Adds a singular chart. Accepts information about the metric to be creating as a chart.
function addChart(metric_name, display_name, explanation, data, category, name_extension, has_range, range, add_tag=true, tags=[], tagOwner, tag_list, data_types){
    if(add_tag) // tags are used for filtering. We don't need tags on the metric_info page.
        addTags(tags, "numeric", metric_name, tagOwner, tag_list, data_types)
    var body = document.getElementById('metric_row');

    var newDiv = document.createElement('div'); // Div containing chart.
    if(add_tag) // Different formatting for the metric_info age vs the other two display pages.
        newDiv.setAttribute("class", category.toLowerCase() + 'Metric Metric col-sm-6 chart-container main-panel');
    else
        newDiv.setAttribute("class", 'MetricPage Metric chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");

    var writing = document.createElement('p'); // Chart header.
    writing.innerHTML = display_name;
    writing.setAttribute("class", "chartHeader");

    var writing2 = document.createElement('p'); // Last seen value of the chart.
    if (typeof(data[data.length-1][metric_name + name_extension]) == 'number')
        writing2.innerHTML = data[data.length -1][ metric_name + name_extension].toFixed(3);
    else
        writing2.innerHTML = "Null"
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");

    var img = document.createElement('img'); // The explanation question mark that provides info on hover.
    if(add_tag){
        img.setAttribute("title", explanation);
        img.setAttribute("src", "/static/img/questionMark.png");
        img.setAttribute("alt", "Learn more about " + metric_name);
        img.setAttribute("class", "learnMore");
    }

    var link = document.createElement('a') // the link to go to metric_info
    link.setAttribute('href', '/learnMore/'+metric_name)
    link.setAttribute('class', 'learnMoreLink')

    var logo = document.createElement('i') // The image for the link
    logo.setAttribute('class', 'fa fa-external-link fa-lg')
    link.appendChild(logo)

    newDiv.appendChild(link) // Add the new objects
    newDiv.appendChild(img);
    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    var removeBtn = document.createElement("button"); // The button that appears when filtering to allow us to delete metrics.
    removeBtn.innerHTML  = "X";
    removeBtn.setAttribute("class", "removeChart");
    removeBtn.setAttribute("style", "display:none");
    removeBtn.setAttribute("onclick", "blackList('" + metric_name + "')");
    newDiv.appendChild(removeBtn);

    var chart = document.createElement('div'); // The morris chart for displaying data.
    chart.id = metric_name;
    chart.setAttribute("class", "morris-chart chartScalerSmall");
    newDiv.appendChild(chart);
    body.appendChild(newDiv);

    var result = createData(data, metric_name + name_extension) // Put data in morris charts format. Can most likely be sped up.
    var chart_data = result[0]
    var chart_descriptions = result[1]

    // Metadata for the morris chart. This is a morris line, all these values are explained in their documentation.
    var myValues  = {
        element: metric_name,
        data: chart_data,
        xkey: 'year',
        descriptions: chart_descriptions,
        ykey: metric_name,
        hideHover: true,
        smooth: false,
        lineColors: ['#000000'],
        parseTime: use_date,
        pointFillColors: ['#000000'],
        ykeys: ['value'],
        labels: ['Value'],
        hoverCallback: function (index, options, content, row) {
                var description = options.descriptions[index];
                return content + "\nDescription: " + description;}
    }

    // We only want to give the morris chart a range if the data says there is a range.
    if(has_range){
        if(range[0] != null){
            myValues['ymin'] = Number(range[0])
        }
        if(range[1] != null){
            myValues['ymax'] = Number(range[1])
        }
        myValues['yLabelFormat'] = function(y){return y.toFixed(2);}
    }
    var morrisLine = new Morris.Line(myValues) // Create morris line with metadata
    graphs[metric_name] = morrisLine; // save it for when we are redoing metrics
}


// Adds a singular boolean chart. Accepts information about the metric to be creating as a chart.
function addBoolChart(metric_name, display_name, explanation, data, category, name_extension, has_range, range, add_tag=true, tags=[], tagOwner, tag_list, data_types){
    if(add_tag) // tags are used for filtering. We don't need tags on the metric_info page.
        addTags(tags, "bool", metric_name, tagOwner, tag_list, data_types)
    var body = document.getElementById('metric_row');


    var newDiv = document.createElement('div'); // Div containing chart.
    if(add_tag) // Different formatting for the metric_info age vs the other two display pages.
        newDiv.setAttribute("class", category.toLowerCase() + 'Metric Metric col-sm-6 chart-container main-panel');
    else
        newDiv.setAttribute("class", 'MetricPage Metric chart-container main-panel');
    newDiv.setAttribute("id", metric_name + "_chart");

    var writing = document.createElement('p');  // Chart header.
    writing.innerHTML = display_name;
    writing.setAttribute("class", "chartHeader");

    var writing2 = document.createElement('p'); // Last seen value of the chart.
    if(data[data.length-1][metric_name + name_extension] == null)
        writing2.innerHTML = "Null"
    else
        writing2.innerHTML = data[data.length -1][ metric_name + name_extension];
    writing2.setAttribute("class", "chartValue");
    writing2.setAttribute("id", metric_name + "LastValue");

    // The explanation question mark that provides info on hover.
    var img = document.createElement('img');
    if(add_tag){
        img.setAttribute("title", explanation);
        img.setAttribute("src", "/static/img/questionMark.png");
        img.setAttribute("alt", "Learn more about " + metric_name);
        img.setAttribute("class", "learnMore");
    }

    var link = document.createElement('a') // the link to go to metric_info
    link.setAttribute('href', '/learnMore/'+metric_name)
    link.setAttribute('class', 'learnMoreLink')

    var logo = document.createElement('i') // The image for the link
    logo.setAttribute('class', 'fa fa-external-link fa-lg')

    link.appendChild(logo)  // Add the new objects
    newDiv.appendChild(link)
    newDiv.appendChild(img);
    newDiv.appendChild(writing);
    newDiv.appendChild(writing2);

    // The button that appears when filtering to allow us to delete metrics.
    var removeBtn = document.createElement("button");
    removeBtn.innerHTML  = "X";
    removeBtn.setAttribute("class", "removeChart");
    removeBtn.setAttribute("style", "display:none");
    removeBtn.setAttribute("onclick", "blackList('" + metric_name + "')");
    newDiv.appendChild(removeBtn);

    // The morris chart for displaying data.
    var chart = document.createElement('div');
    chart.setAttribute("class", "overflow_table")
    chart.id = metric_name;
    chart.setAttribute("class", "morris-chart chartScalerSmall");
    newDiv.appendChild(chart);
    body.appendChild(newDiv);

    var result = createBoolData(data, metric_name + name_extension) // Put data in morris charts format.
    var chart_data = result[0]
    var chart_descriptions = result[1]

    // Metadata for the morris chart. This is a morris line, all these values are explained in their documentation.
    var myValues  = {
        element: metric_name,
        data: chart_data,
        xkey: 'year',
        descriptions: chart_descriptions,
        ykey: metric_name,
        hideHover: true,
        smooth: false,
        lineColors: ['#000000'],
        parseTime: use_date,
        pointFillColors: ['#000000'],
        ykeys: ['value'],
        labels: ['Value'],
        hoverCallback: function (index, options, content, row) {
                var description = options.descriptions[index];
                return content + "\nDescription: " + description;}
    }
    // We only want to give the morris chart a range if the data says there is a range.
    if(has_range){
        if(range[0] != null){
            myValues['ymin'] = Number(range[0])
        }
        if(range[1] != null){
            myValues['ymax'] = Number(range[1])
        }
    }

    var morrisLine = new Morris.Line(myValues) // Create the morris chart with the metadata.
    bool_charts[metric_name] = morrisLine; // Save it for quick access when redoing metrics.
}


// Adds a singular table. Accepts information about the metric to be creating as a chart.
function addTable(metric_name, display_name, explanation, data, category, optionalName="", optionalNumber="", add_tag=false, tags=[], tagOwner, tag_list, data_types){
    if(add_tag) // tags are used for filtering. We don't need tags on the metric_info page.
        addTags(tags, "matrix", metric_name, tagOwner, tag_list, data_types)
    var body = document.getElementById('metric_row');

    var newDiv = document.createElement('div'); // Div containing table
    if(add_tag)
        newDiv.setAttribute("class", category.toLowerCase() + 'Metric Metric col-sm-6 chart-container main-panel ');
    else
        newDiv.setAttribute("class", 'MetricPage Metric chart-container main-panel');

    // Special values for displaying vector dicts. We need unique ID's and use their index number to do so.
    if(optionalNumber!="")
        optionalNumber = "|"+optionalNumber;
    newDiv.setAttribute("id", metric_name + "_chart"+optionalNumber);

    var writing = document.createElement('p');  // Chart header.
    writing.innerHTML = display_name;
    if(optionalName != "") // Option to also add a string to display, for example to show a feature name with vector dicts
        writing.innerHTML += " - " + optionalName
    writing.setAttribute("class", "chartHeader");

    var img = document.createElement('img'); // The explanation question mark that provides info on hover.
    var link = document.createElement('a') // Link to metric_info page
    var logo = document.createElement('i') // Logo for the link
    if(add_tag){
        img.setAttribute("title", explanation);
        img.setAttribute("src", "/static/img/questionMark.png");
        img.setAttribute("alt", "Learn more about " + metric_name);
        img.setAttribute("class", "learnMore");

        link.setAttribute('href', '/learnMore/'+metric_name)
        link.setAttribute('class', 'learnMoreLink')

        logo.setAttribute('class', 'fa fa-external-link fa-lg')
    }
    // Add the new objects
    newDiv.appendChild(img);
    newDiv.appendChild(writing);

    // The button that appears when filtering to allow us to delete metrics.
    var removeBtn = document.createElement("button");
    removeBtn.innerHTML  = "X";
    removeBtn.setAttribute("class", "removeChart");
    removeBtn.setAttribute("style", "display:none");
    removeBtn.setAttribute("onclick", "blackList('" + metric_name + "')");
    newDiv.appendChild(removeBtn);

    link.appendChild(logo)
    newDiv.appendChild(link)

    var chart = document.createElement('div'); // The table container for displaying data.
    chart.setAttribute("class", "overflow_table")
    chart.id = metric_name;
    newDiv.appendChild(chart);
    body.appendChild(newDiv);

    var table = generateTableFromArray(data) // Generate a table using our function from an array/2d array.
    chart.appendChild(table);
    matrices[metric_name] = chart; // Store so we can quickly redo metrics.
}


// Adds a singular vector dict. Current assumption is that they are paired to the features of the dataset, so value 0 applies to feature 0.
function addVectorDict(metric_name, display_name, explanation, features, data, category, add_tag, tags, tagOwner, tag_list, data_types){
    var curData = data[data.length -1][metric_name]
    if(curData == null)
        return
    var result = {}
    for(var i = 0; i<curData.length; i++){
        if(curData[i] != null){
            var table = dict_to_table(curData[i]); // Create a table for each dictionary.
            addTable(metric_name, display_name, explanation, table, category, features[i], String(i), add_tag, tags, tagOwner, tag_list, data_types);
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

function addTags(metric_tags, metric_type, metric_name, tagOwner, tag_list, data_types){
    for (var i = 0; i < metric_tags.length; i++){
        if (tag_list[metric_tags[i]] == null)
            tag_list[metric_tags[i]] = []
        tag_list[metric_tags[i]].push(metric_name);
    }

    for(var i = 0; i<categories.length; i++){
        if(metric_tags.includes(categories[i])){
            for(var j = 0; j<metric_tags.length; j++){
                if(metric_tags[j] != categories[i] && !tagOwner[categories[i]].includes(metric_tags[j]))
                    tagOwner[categories[i]].push(metric_tags[j])
           }
        }
    }
    if (data_types[metric_type] == null){
        data_types[metric_type] = []
    }
    data_types[metric_type].push(metric_name)
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
                    year: String(i),
                    value: value
                });
            }
            descriptions.push(data[i]["metadata > description"])
        }
    }
    return [ret, descriptions];
}



function createMultiCategoryBoxes(metrics, tagOwner, data_types){
    for (var cate in tagOwner)
        createCategoryBoxes(metrics, cate, tagOwner[cate])
    createDatatypeBoxes(metrics, data_types)
    createBoxButton()
}


function createBoxes(metrics, tags, category, data_types){
    createCategoryBoxes(metrics, category, tags)
    createDatatypeBoxes(metrics, data_types)
    createBoxButton()
}



function createCategoryBoxes(metrics, category, tags){
    var body = document.getElementById('tag_selection');
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
    for (var i = 0; i<tags.length; i++) {
        var group = tags[i]
        var newBox = document.createElement("input");
        newBox.setAttribute("type", "checkbox");
        newBox.setAttribute("id", group);
        newBox.setAttribute("value", true);
        newBox.setAttribute("name", group);
        newBox.setAttribute("class", "selectorBox");
        newBox.setAttribute("class", category.toString().toLowerCase() + "Box" + " innerBox");
        newBox.setAttribute("checked", true);
        newBox.setAttribute("onclick", "checkParent(this, '" + category + "')");
        var label = document.createElement("label");
        label.setAttribute("for", group + category.toString().toLowerCase());
        label.innerHTML = group;
        var br = document.createElement("br")
        body.appendChild(newBox);
        body.appendChild(label);
        body.appendChild(br);
    }
}


function createDatatypeBoxes(metrics, data_types){
    var body = document.getElementById("datatype_selection")
    for (var i in data_types) {
        var newBox = document.createElement("input");
        newBox.setAttribute("type", "checkbox");
        newBox.setAttribute("id", i);
        newBox.setAttribute("value", true);
        newBox.setAttribute("name", i);
        newBox.setAttribute("class", "selectorBox");
        newBox.setAttribute("class", i + "_Box");
        newBox.setAttribute("checked", true);
        newBox.setAttribute("onclick", "updateDatatypes()");
        var label = document.createElement("label");
        label.setAttribute("for", i);
        label.innerHTML = i;
        var br = document.createElement("br")
        body.appendChild(newBox);
        body.appendChild(label);
        body.appendChild(br);
    }
}

function createBoxButton(){
    var button = document.createElement("button");
    button.setAttribute("class", "selectorButton");
    button.innerHTML = "Done";
    button.setAttribute("onclick", "doneEdit();")
    document.getElementById("selector").appendChild(button);
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
function doneEdit() {
    document.getElementById("selector").style.display = "none";
    var list = document.getElementsByClassName("removeChart");
    for (var i = 0; i < list.length; i++)
        list[i].style.display = "none";
}

// Generates the white list (what metrics should be shown, based on category) by looking at the checked boxes
function generateWhiteList() {
    whitelist = [];
    data_types = []
    var datatypeBox = document.getElementById("datatype_selection")
    var boxes = datatypeBox.getElementsByTagName("input")
    for (var i = 0; i<boxes.length; i++){
        if (boxes[i].checked){
            data_types.push(boxes[i].id.toString())
        }
    }

    var boxes = document.getElementById("tag_selection").getElementsByTagName("input");
    for (var i = 0; i < boxes.length; i++) {
        if (boxes[i].checked) {
            var id = boxes[i].id.toString();
            if(id.indexOf("_mainBox") == -1){
                var list = tags[id];
                for (var j = 0; j < list.length; j++) {
                    if(data_types.includes(metric_info[list[j]]["type"])){
                        whitelist.push(list[j]);
                    }
                }
            }
        }
    }
    displayWhiteList()
}

// Display the white listed metrics
function displayWhiteList() {
    var row = document.getElementById("metric_row");
    var divs = row.getElementsByClassName("Metric");
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


// Fill graphs with new data
function redoMetrics2(data) {
    for (var type in graphs) {
        var ext = "";
        if(metric_info[type]["type"] == "vector")
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
        var result = createBoolData(data, type + ext);
        var new_data = result[0]
        var newExplanations = result[1]
        bool_charts[type].options.parseTime = use_date
        bool_charts[type].setData(new_data);
        bool_charts[type].options.descriptions = newExplanations
    }
    for (var type in matrices){
        if(metric_info[type]['type'] == 'matrix' || metric_info[type]['type'] == 'vector'){
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
        if(metric_info[type]['type'] == 'vector-dict'){
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


// Checks parent status in checkboxes, once all child checks are unchecked, or one is checked.
function updateDatatypes(type) {
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
            if ((text.toLowerCase().indexOf(filter) > -1 || mustInclude.includes(hiddenText)) && whitelist.includes(hiddenText) && !blacklist.includes(hiddenText)) {
                divs[i].style.display = "";
            } else {
                divs[i].style.display = "none";
            }
        }
    }
}

function date_slider(){
    var slider = document.getElementById('slider_input')
    use_date = !slider.checked;
    redoMetrics()
}


function view_slider(){
    var slider = document.getElementById('view_input')
    use_date = !slider.checked;
    var style = ""
    var svg_style = ""
    var header_style = ""
    var text_display = ""
    var chart_scaler = ""
    if(!slider.checked){
        style = 'width: 32%; margin-left: 1%; margin-top: 2%; fill: black; height: 290px;'
        header_style = "text-align: center; font-size: 25px; margin-top: 10px; margin-bottom: 0px; color: black;"
        svg_style = "width:100%;"
        text_display = "display:block; margin-left:0px; text-align: center; font-size: 25px; margin-top: 5px; margin-bottom: 0px; color: black;"
        chart_scaler = "height:60%;"
    }
    var row = document.getElementById("metric_row");
    var boxes = row.getElementsByClassName("Metric");
    for(var i = 0; i<boxes.length; i++){
        boxes[i].setAttribute("style", style);
        var svgs = boxes[i].getElementsByTagName("svg")
        if(svgs.length > 0)
            svgs[0].setAttribute("width", "100%")

        var text = boxes[i].getElementsByClassName("chartValue")
        if(text.length > 0)
            text[0].setAttribute("style", text_display)

        var graph = boxes[i].getElementsByClassName("morris-chart")
        if(graph.length > 0)
            graph[0].setAttribute("style", chart_scaler)
    }
    var texts = row.getElementsByClassName("chartHeader");
    for(var i = 0; i<texts.length; i++){
        texts[i].setAttribute("style", header_style);
    }

    for(var chart in graphs){
        if(document.getElementById(chart+"LastValue").innerHTML != "Null")
                graphs[chart].redraw()
    }
    for(var chart in bool_charts){
        if(document.getElementById(chart+"LastValue").innerHTML != "Null"){
            bool_charts[chart].redraw()
        }
    }
}
