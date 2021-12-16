'use strict';

// GLOBAL VARIABLES.
// Note, these can be viewed cross file.

// Stores the charts created on screen
var graphs = {};
var matrices = {};
var bool_charts = {}
var dict_charts = {}

// Stores data received from backend about metrics and model
var metrics;
var metric_info;
var model_info;
var metric_data

// Information about metric tags
var tags = {}
var data_types = []
var tagOwner = {'fairness': [], 'performance': [], 'robustness': [], 'stats': []}
var categories = ['fairness', 'performance', 'robustness', 'stats']

// Information about what to display.
var use_date = false;
var page_ready = false;
var whitelist = []; // contains all metrics whose category is checked
var blacklist = []; // contains which metrics were X'd out


// Run check_data function once a second.
$(document).ready(function() {
        setInterval("check_data()", 1000); // call every second
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


// Function called from view_class which causes everything to load and display when it opens.
// We start by loading the list of all metrics.
function loadAll() {
    page_ready = false
    fetch('/getMetricList').then(function (response) {
        return response.json();
    }).then(function(text){
        metrics = text
        loadExplanations(text);
    });
}



// Get metric Metadata.
function loadExplanations(metrics) {
    fetch('/getMetricInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        metric_info = text;
        load_model_info(metrics, text);
    });
}


// Get Model info
function load_model_info(metrics, explanations) {
    fetch('/getModelInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        model_info = text;
        load_data(metrics, explanations);
    });
}


// Get the metric values.
function load_data(metrics, data, modelInfo) {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getData/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            metric_data = text
            callAllFunctions(metrics, data, text);
        });
}


// Calls functions from helper_functions.js. This will use the data we got to display our metrics.
function callAllFunctions(metrics, explanations, data) {
    for(var i = 0; i<categories.length; i++){ // Draw the graphs
        createMetrics(metrics, metric_info, explanations, data, categories[i], true);
    }
    createMultiCategoryBoxes(metrics); // Create the filtering boxes
    createWhiteList(metrics); // Setup filtering functionality
    page_ready = true
}