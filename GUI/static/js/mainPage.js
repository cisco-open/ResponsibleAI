'use strict';

var graphs = {};
var metrics;
var page_ready = false;
var use_date= true;



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
function createData(data, key) {
    var ret = [];
    var descriptions = []
    for (var i = 0; i < data.length; i++) {
        if(data[i]["metadata"]["scores"][key] != null){
            if(use_date){
                ret.push({
                    year: data[i]["metadata"]["date"]["value"],
                    value: (100* data[i]["metadata"]["scores"][key][0] / data[i]["metadata"]["scores"][key][1]).toFixed(1)
                });
            }
            else{
                ret.push({
                    year: data[i]["metadata"]["description"]["value"],
                    value: (100* data[i]["metadata"]["scores"][key][0] / data[i]["metadata"]["scores"][key][1]).toFixed(1)
                });
            }
            descriptions.push(data[i]["metadata"]["description"]["value"])
        }
    }
    return [ret, descriptions];
}



function createMetrics(data, explanations) {
    var divs = ['fairness', 'robust', 'performance', 'explainability'];
    var names = ["Fairness", "Robustness", "Performance", "Explainability"];
    var explanations = {"fairness": {"name": "fairness", "explanation": "Measures how fair a model's predictions are.", "display_name": "Fairness"}, "robust": {"name": "robustness", "explanation": "Measures a model's resiliance to time and sway.", "display_name": "Robustness"}, "explainability": {"name": "explainability", "explanation": "Measures how explainable the model is.", "display_name": "Explainability"}, "performance": {"name": "performance", "explanation": "Performance describes how well at predicting the model was.", "display_name": "Performance"}}
    for (var j in divs) {
        var i = divs[j]
        var result = createData(data, i);
        var new_data = result[0]
        var chart_explanations = result[1]
        var morrisLine = new Morris.Line({
            element: i + "Chart",
            data: new_data,
            descriptions: chart_explanations,
            xkey: 'year',
            ykey: i,
            ymax: 100,
            ymin: 0,
            parseTime: true,
            hideHover: true,
            lineColors: ['#000000'],
            pointFillColors: ['#000000'],
            ykeys: ['value'],
            labels: ['Value'],
            hoverCallback: function (index, options, content, row) {
                var description = options.descriptions[index];
                return content + "\nDescription: " + description;}
        });
        graphs[i] = morrisLine;
        var img = document.getElementById(i + "KnobQ");
        img.setAttribute("title", explanations[i]["explanation"]);

        var circle = document.getElementById(i + "Circle");
        circle.setAttribute("stroke-dasharray",
            (data[data.length - 1]['metadata']['scores'][i][0] / data[data.length - 1]['metadata']['scores'][i][1] * 100).toFixed(0) + ", 100");
        var circleText = document.getElementById(i + "Text");

        circleText.innerHTML = (data[data.length - 1]['metadata']['scores'][i][0] / data[data.length - 1]['metadata']['scores'][i][1] * 100).toFixed(1) + "%";
        // console.log("Setting value for " + i + " to " + new_data[new_data.length - 1]['value'].toFixed(1))

        // var percentage = new_data[new_data.length - 1]['value']/Object.keys(data[data.length-1][i]['list']).length).toFixed(2)*100
    }
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
        var result = createData(data, type);
        var new_data = result[0]
        var newExplanations = result[1]
        graphs[type]['options'].parseTime = use_date
        graphs[type].setData(new_data);
        graphs[type].options.descriptions = newExplanations
        var myValue = 0
        if (new_data.length >= 1)
            myValue = parseFloat(new_data[new_data.length - 1]['value'])
        var circle = document.getElementById(type + "Circle");
        circle.setAttribute("stroke-dasharray", myValue.toFixed(0) + ", 100");
        var circleText = document.getElementById(type + "Text");
        circleText.innerHTML = myValue.toFixed(1) + "%";
    }
}



function date_slider(){
    var slider = document.getElementById('slider_input')
    use_date = !slider.checked;
    redoMetrics()
}

