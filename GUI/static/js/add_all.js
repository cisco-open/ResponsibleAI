'use strict';

var whitelist = []; // contains all metrics whose category is checked
var blacklist = []; // contains which metrics were X'd out
var metrics; // global variable contains all metrics on screen
var graphs = {}; // list of all graphs
var matrices = {};
var bool_charts = {}
var dict_charts = {}
var metric_info;
var model_info;
var tags = {}
var data_types = []
var tagOwner = {'fairness': [], 'performance': [], 'robustness': [], 'stats': []}
var categories = ['fairness', 'performance', 'robustness', 'stats']
var metric_data
var use_date = false;
var page_ready = false;


$(document).ready(function() {
        setInterval("check_data()", 1000); // call every second
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


// loads metrics. Contains lists of metrics, and metric tags.
function loadAll() {
    page_ready = false
    fetch('/getMetricList').then(function (response) {
        return response.json();
    }).then(function(text){
        metrics = text
        loadExplanations(text);
    });
}



// Loads explanations.
function loadExplanations(metrics) {
    fetch('/getMetricInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        metric_info = text;
        load_model_info(metrics, text);
    });
}


// Loads explanations.
function load_model_info(metrics, explanations) {
    fetch('/getModelInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        model_info = text;
        load_data(metrics, explanations);
    });
}


// Queries Data
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


// Use collected data to create metrics, boxes and white list metrics by category
function callAllFunctions(metrics, data, df_json) {
    for(var i = 0; i<categories.length; i++){
        createMetrics(metrics, metric_info, data, df_json, categories[i], true, tagOwner, tags, data_types);
    }
    createMultiCategoryBoxes(metrics, tagOwner, data_types);
    createWhiteList(metrics);
    page_ready = true
}