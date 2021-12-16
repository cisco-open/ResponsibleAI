'use strict';


// GLOBAL VARIABLES

// Chart and graph storage
var graphs = {};
var main_chart = null;

// Data caching
var metrics;
var metric_values = null

// Display options
var page_ready = false;
var use_date= false;



// Run once a second
$(document).ready(function() {
        setInterval("check_data()", 1000); // call every 10 seconds
});

// Ping back end to see if there are any updates
function check_data() {
    if(page_ready){
       fetch('/updateCertificates').then(function (response) {
            return response.json();
        }).then(function(result){
            if (result){
                redoMetrics();
            }
        });
    }
}


// Load in the certificate data
function load_data() {
    page_ready = false
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getCertification/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            load_explanations(text);
        });
}


// Get the explanations for certificates
function load_explanations(data) {
    return fetch('/getCertificationMeta')
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            load_metric_data(data, text);
        });
}


// Get the metric values
function load_metric_data(data, text) {  // Needed because we are displaying balanced accuracy for performance.
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (new_text) {
            metric_values = new_text
            createMetrics(data, text);
        });
}


// Used to create the main page data for morris charts.
// Specialized function, each point consists of a time stamp or description and the score for each of the 4 categories.
function createData(data) {
    var ret = [];
    var keys = ["fairness", "robustness", "performance", "explainability"]
    var descriptions = []
    var names = []

    console.log()
    console.log(metric_values)

    for (var i = 0; i < data.length; i++) {
        if(data[i]["metadata"]["scores"]["performance"] != null){
            if(use_date){
                ret.push({
                    year: data[i]["metadata"]["date"]["value"],
                    name: data[i]["metadata"]["date"]["value"],
                    fairness: (100* data[i]["metadata"]["scores"]["fairness"][0] / data[i]["metadata"]["scores"]["fairness"][1]).toFixed(1),
                    performance: (100* metric_values[i]['performance_cl > balanced_accuracy']).toFixed(1),
                    explainability: (100* data[i]["metadata"]["scores"]["explainability"][0] / data[i]["metadata"]["scores"]["explainability"][1]).toFixed(1),
                    robustness: (100* data[i]["metadata"]["scores"]["robustness"][0] / data[i]["metadata"]["scores"]["robustness"][1]).toFixed(1)
                });
            }
            else{
                ret.push({
                    year: i,
                    name: data[i]["metadata"]["description"]["value"],
                    fairness: (100* data[i]["metadata"]["scores"]["fairness"][0] / data[i]["metadata"]["scores"]["fairness"][1]).toFixed(1),
                    performance: (100* metric_values[i]['performance_cl > balanced_accuracy']).toFixed(1),
                    explainability: (100* data[i]["metadata"]["scores"]["explainability"][0] / data[i]["metadata"]["scores"]["explainability"][1]).toFixed(1),
                    robustness: (100* data[i]["metadata"]["scores"]["robustness"][0] / data[i]["metadata"]["scores"]["robustness"][1]).toFixed(1)
                });
            }
            descriptions.push(data[i]["metadata"]["description"]["value"])
            names.push(data[i]["metadata"]["description"]["value"])
        }
    }
    return [ret, descriptions, names];
}


// Create all displays on the front page
function createMetrics(data, explanations) {
    var divs = ['fairness', 'robustness', 'performance', 'explainability'];
    var names = ["Fairness", "Robustness", "Performance", "Explainability"];
    var explanations = {"fairness": {"name": "fairness", "explanation": "Measures how fair a model's predictions are.", "display_name": "Fairness"}, "robustness": {"name": "robustness", "explanation": "Measures a model's resiliance to time and sway.", "display_name": "Robustness"}, "explainability": {"name": "explainability", "explanation": "Measures how explainable the model is.", "display_name": "Explainability"}, "performance": {"name": "performance", "explanation": "Performance describes how well at predicting the model was.", "display_name": "Performance"}}

    for (var j in divs) { // for each category, create the knob for it
        var i = divs[j]
        graphs[i] = i;
        var img = document.getElementById(i + "KnobQ");
        img.setAttribute("title", explanations[i]["explanation"]);
        var circle = document.getElementById(i + "Circle");
        circle.setAttribute("stroke-dasharray",
            (data[data.length - 1]['metadata']['scores'][i][0] / data[data.length - 1]['metadata']['scores'][i][1] * 100).toFixed(0) + ", 100");
        var circleText = document.getElementById(i + "Text");
        circleText.innerHTML = (data[data.length - 1]['metadata']['scores'][i][0] / data[data.length - 1]['metadata']['scores'][i][1] * 100).toFixed(1) + "%";
    }

    // Repeat this for performance, which has to be done separately as we are using a metric value for it instead of a certificate.
    var i = "performance"
    graphs[i] = i;
    var img = document.getElementById(i + "KnobQ");
    img.setAttribute("title", explanations[i]["explanation"]);
    var circle = document.getElementById(i + "Circle");
    circle.setAttribute("stroke-dasharray",
        (100* metric_values[metric_values.length-1]['performance_cl > balanced_accuracy']).toFixed(0) + ", 100");
    var circleText = document.getElementById(i + "Text");
    circleText.innerHTML =(100* metric_values[metric_values.length-1]['performance_cl > balanced_accuracy']).toFixed(1) + "%";


    // Create the really big chart
    var result = createData(data)
    var all_data = result[0]
    var descriptions = result[1]

    var chart_explanations = result[1]
    var names = result[2]

    var events = [0, 0]
    if(!use_date){
        var range = (all_data[all_data.length-1]['year'] - all_data[0]['year'])*0.01
        events[0] = all_data[0]['year'] - range
        events[1] = all_data[all_data.length-1]['year'] + range
        console.log(events)
    }

    // Chart metadata
    var options = {
        element: "allChart",
        data: all_data,
        descriptions: descriptions,
        xkey: 'year',
        ykey: i,
        names: names,
        events: events,
        eventStrokeWidth:0,
        ymax: 100,
        ymin: 0,
        parseTime: use_date,
        hideHover: true,
        lineColors: ['red', 'green', 'blue', 'yellow'],
        pointFillColors: ['red', 'green', 'blue', 'yellow'],
        ykeys: ['fairness', 'robustness', 'performance', 'explainability'],
        labels: ['fairness', 'robustness', 'performance', 'explainability'],
        hoverCallback: function (index, options, content, row) {
            var description = options.descriptions[index];
            return content + "\nDescription: " + description;},
        xLabelFormat: function(index){
            if (index.src != null)
                return index.src.name
            return index
        },
    }
    var morrisLine = new Morris.Line(options);
    main_chart = morrisLine
    page_ready = true
}


// Start function for redoing the page, recollects data.
function redoMetrics() {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    page_ready = false
    return fetch('/getCertification/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            load_metric_data_redo(text)
            page_ready = true
        });
}


// Get the metric data again, for balanced accuracy
function load_metric_data_redo(text) {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (new_text) {
            metric_values = new_text
            redoMetrics2(text);
        });
}


// Reset the knobs and redo the big chart
function redoMetrics2(data) {
    for (var type in graphs) { // reset the charts.
        if(type == "performance")
            continue;
        var myValue = 0
        if (data.length >= 1)
            myValue = data[data.length - 1]['metadata']['scores'][type][0] / data[data.length - 1]['metadata']['scores'][type][1] * 100
        var circle = document.getElementById(type + "Circle");
        circle.setAttribute("stroke-dasharray", myValue.toFixed(0) + ", 100");
        var circleText = document.getElementById(type + "Text");
        circleText.innerHTML = myValue.toFixed(1) + "%";
    }
    var i = "performance" // performance is done seperately
    var circle = document.getElementById(i + "Circle");
    circle.setAttribute("stroke-dasharray",
        (100* metric_values[metric_values.length-1]['performance_cl > balanced_accuracy']).toFixed(0) + ", 100");
    var circleText = document.getElementById(i + "Text");
    circleText.innerHTML =(100* metric_values[metric_values.length-1]['performance_cl > balanced_accuracy']).toFixed(1) + "%";


    // Redo the main chart by recreating the data and setting the data of the chart
    var result = createData(data);
    var new_data = result[0]
    var newExplanations = result[1]
    main_chart['options'].parseTime = use_date
    main_chart.options.events = [new_data[0].year, new_data[new_data.length-1].year]
    main_chart.setData(new_data);
    main_chart.options.descriptions = newExplanations
}


// Alternate X axis from either date or index/description.
function date_slider(){
    var slider = document.getElementById('slider_input')
    use_date = !slider.checked;
    redoMetrics()
}

