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
var use_date = false;
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
    var metric_info = {}
    var range = JSON.parse(metric_range.replace("None", "null"))
     metric_info[metric_name] = {"display_name": metric_display_name, "has_range": metric_has_range,
            "range":range, "tags": []}
    var explanations = {}
    explanations[metric_name] = {"explanation": null}


    createMetric(metric_name, metric_type, metric_info, explanations, data, "bias", false);
    page_ready = true
}

