'use strict';

var graphs = {};
var metrics;
var page_ready = false;
var use_date= true;


/*
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

*/

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

    for (var i = 0; i < data.length; i++) {
        ret.push({
            year: data[i]["metadata"]["date"],
            value: (data[i][key]["score"] / Object.keys(data[i][key]['list']).length).toFixed(2)*100
        });
    }
    return ret;
}


function createData(data, key) {
    var ret = [];
    for (var i = 0; i < data.length; i++) {
        ret.push({
            year: data[i]["metadata"]["date"],
            value: data[i][key]["score"]
        });
    }
    return ret;
}



function createMetrics(data, explanations) {
    var divs = ['fairness', 'robust', 'performance', 'explainability'];
    var names = ["Fairness", "Robustness", "Performance", "Explainability"];
    for (var i in explanations) {
        var new_data = createData(data, i);
        console.log("NEW DATA: " + JSON.stringify(new_data))
        var morrisLine = new Morris.Line({
            element: i + "Chart",
            data: new_data,
            xkey: 'year',
            ykey: i,
            ymax: 100,
            ymin: 0,
            hideHover: true,
            lineColors: ['#000000'],
            pointFillColors: ['#000000'],
            ykeys: ['value'],
            labels: ['Value']
        });
        graphs[i] = morrisLine;
        var img = document.getElementById(i + "KnobQ");
        img.setAttribute("title", explanations[i]["explanation"]);

        var circle = document.getElementById(i + "Circle");
        circle.setAttribute("stroke-dasharray", new_data[new_data.length - 1]['value'].toFixed(0) + ", 100");
        var circleText = document.getElementById(i + "Text");

        circleText.innerHTML = new_data[new_data.length - 1]['value'].toFixed(1) + "%";
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
        var new_data = createData(data, type);
        graphs[type]['options'].parseTime = use_date
        graphs[type].setData(new_data);
        graphs[type].options.descriptions = newExplanations

        var circle = document.getElementById(type + "Circle");
        circle.setAttribute("stroke-dasharray", new_data[new_data.length - 1]['value'].toFixed(0) + ", 100");
        var circleText = document.getElementById(type + "Text");
        circleText.innerHTML = new_data[new_data.length - 1]['value'].toFixed(1) + "%";
    }
}



function date_slider(){
    var slider = document.getElementById('slider_input')
    use_date = !slider.checked;
    redoMetrics()
}

