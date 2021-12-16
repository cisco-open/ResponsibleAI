'use strict';

// Stores the charts created on screen
var graphs = {}; // list of all graphs
var matrices = {};
var bool_charts = {}
var dict_charts = {}

// Stores data received from backend about metrics and model
var metrics; // global variable contains all metrics on screen
var metric_info;
var model_info;

// Information about metric tags
var tags = {}
var tagOwner = {'fairness': [], 'performance': [], 'robustness': [], 'stats': []}
var categories = ['fairness', 'performance', 'robustness', 'stats']
var data_types = []

// Information about what to display.
var use_date= false;
var page_ready = false;
var whitelist = []; // contains all metrics whose category is checked
var blacklist = []; // contains which metrics were X'd out


// Run check_data function once a second.
$(document).ready(function() {
        setInterval("check_data()", 1000); // call every 10 seconds
});

// Pings the back end if data was arrived once a second, if so reload graphs.
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

// loads metrics. Contains lists of metrics, and metric tags.
function loadMetrics(category) {
    page_ready = false
    fetch('/getMetricList').then(function (response) {
        return response.json();
    }).then(function(text){
        metrics = text
        loadExplanations(text, category);
    });
}

// Get metric metadata.
function loadExplanations(metrics, category) {
    fetch('/getMetricInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        metric_info = text;
        load_model_info(metrics, text, category);
    });
}


// Loads model information
function load_model_info(metrics, explanations, category) {
    fetch('/getModelInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        model_info = text;
        load_data(metrics, explanations, category);
    });
}

// Load data to display
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


// Calls functions from helper_functions.js. This will use the data we got to display our metrics.
function callAllFunctions(metrics, data, df_json, category) {
    createMetrics(metrics, metric_info, data, df_json, category, true, tagOwner, tags, data_types);
    createBoxes(metrics, tags, category);
    createWhiteList(metrics, category);
    page_ready = true
}



