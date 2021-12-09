'use strict';

var graphs = {};
var metrics;
var page_ready = false;
var use_date= true;
var main_chart = null;


$(document).ready(function() {
        setInterval("check_data()", 1000); // call every 10 seconds
});

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


function load_explanations(data) {
    return fetch('/getCertificationMeta')
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            createMetrics(data, text);
        });
}



// Used to create the data for the morris chart
function createData(data) {
    var ret = [];
    var keys = ["fairness", "robustness", "performance", "explainability"]
    var descriptions = []
    for (var i = 0; i < data.length; i++) {
        if(data[i]["metadata"]["scores"]["performance"] != null){
            if(use_date){
                ret.push({
                    year: data[i]["metadata"]["date"]["value"],
                    fairness: (100* data[i]["metadata"]["scores"]["fairness"][0] / data[i]["metadata"]["scores"]["fairness"][1]).toFixed(1),
                    performance: (100* data[i]["metadata"]["scores"]["performance"][0] / data[i]["metadata"]["scores"]["performance"][1]).toFixed(1),
                    explainability: (100* data[i]["metadata"]["scores"]["explainability"][0] / data[i]["metadata"]["scores"]["explainability"][1]).toFixed(1),
                    robustness: (100* data[i]["metadata"]["scores"]["robustness"][0] / data[i]["metadata"]["scores"]["robustness"][1]).toFixed(1)
                });
            }
            else{
                ret.push({
                    year: data[i]["metadata"]["description"]["value"],
                    fairness: (100* data[i]["metadata"]["scores"]["fairness"][0] / data[i]["metadata"]["scores"]["fairness"][1]).toFixed(1),
                    performance: (100* data[i]["metadata"]["scores"]["performance"][0] / data[i]["metadata"]["scores"]["performance"][1]).toFixed(1),
                    explainability: (100* data[i]["metadata"]["scores"]["explainability"][0] / data[i]["metadata"]["scores"]["explainability"][1]).toFixed(1),
                    robustness: (100* data[i]["metadata"]["scores"]["robustness"][0] / data[i]["metadata"]["scores"]["robustness"][1]).toFixed(1)
                });
            }
            descriptions.push(data[i]["metadata"]["description"]["value"])
        }
    }
    return [ret, descriptions];
}



function createMetrics(data, explanations) {
    var divs = ['fairness', 'robustness', 'performance', 'explainability'];
    var names = ["Fairness", "Robustness", "Performance", "Explainability"];
    var explanations = {"fairness": {"name": "fairness", "explanation": "Measures how fair a model's predictions are.", "display_name": "Fairness"}, "robustness": {"name": "robustness", "explanation": "Measures a model's resiliance to time and sway.", "display_name": "Robustness"}, "explainability": {"name": "explainability", "explanation": "Measures how explainable the model is.", "display_name": "Explainability"}, "performance": {"name": "performance", "explanation": "Performance describes how well at predicting the model was.", "display_name": "Performance"}}

    for (var j in divs) {
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

    var result = createData(data)
    var all_data = result[0]
    var descriptions = result[1]
    /*
    all_data = [{year: '2021-12-08 09:10:37', fairness: '30', robustness: '20', performance: "60", explainability: "80"},
            {year: '2021-12-09 09:10:37', fairness: '40', robustness: '10', performance: "80", explainability: "85"}
    ]
    */
    var chart_explanations = result[1]
    var options = {
        element: "allChart",
        data: all_data,
        descriptions: descriptions,
        xkey: 'year',
        ykey: i,
        ymax: 100,
        ymin: 0,
        parseTime: true,
        hideHover: true,
        lineColors: ['red', 'green', 'blue', 'yellow'],
        pointFillColors: ['red', 'green', 'blue', 'yellow'],
        ykeys: ['fairness', 'robustness', 'performance', 'explainability'],
        labels: ['fairness', 'robustness', 'performance', 'explainability'],
        parseTime: true,
        hoverCallback: function (index, options, content, row) {
            var description = options.descriptions[index];
            return content + "\nDescription: " + description;},
    }
    var morrisLine = new Morris.Line(options);
    main_chart = morrisLine
    page_ready = true
}


function redoMetrics() {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    page_ready = false
    return fetch('/getCertification/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            redoMetrics2(text)
            page_ready = true
        });
}

function redoMetrics2(data) {
    for (var type in graphs) {
        var myValue = 0
        if (data.length >= 1)
            myValue = data[data.length - 1]['metadata']['scores'][type][0] / data[data.length - 1]['metadata']['scores'][type][1] * 100
        var circle = document.getElementById(type + "Circle");
        circle.setAttribute("stroke-dasharray", myValue.toFixed(0) + ", 100");
        var circleText = document.getElementById(type + "Text");
        circleText.innerHTML = myValue.toFixed(1) + "%";
    }
    var result = createData(data);
    var new_data = result[0]
    var newExplanations = result[1]
    main_chart['options'].parseTime = use_date
    main_chart.setData(new_data);
    main_chart.options.descriptions = newExplanations
}



function date_slider(){
    var slider = document.getElementById('slider_input')
    use_date = !slider.checked;
    redoMetrics()
}

