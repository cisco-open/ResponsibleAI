'use strict';

var metric_file = "output/metric_list.json";
var explanation_file = "/Flask-Admin-Dashboard/static/output/metric_info.json";

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
var tagOwner = {'fairness': [], 'performance': [], 'robustness': [], 'stats': []}
var categories = ['fairness', 'performance', 'robustness', 'stats']
var data_types = []
var use_date= false;
var page_ready = false;


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


function loadExplanations(metrics, category) {
    fetch('/getMetricInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        metric_info = text;
        load_model_info(metrics, text, category);
    });
}


// Loads explanations.
function load_model_info(metrics, explanations, category) {
    fetch('/getModelInfo').then(function (response) {
        return response.json();
    }).then(function(text){
        model_info = text;
        load_data(metrics, explanations, category);
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
    createMetrics(metrics, metric_info, data, df_json, category, true, tagOwner, tags, data_types);
    createBoxes(metrics, tags, category);
    createWhiteList(metrics, category);
    page_ready = true
}



