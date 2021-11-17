'use strict';

var graphs = {};
var metrics;


function load_data() {
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
            year: data[i]["date"],
            value: data[i][key]
        });
    }
    return ret;
}


function createMetrics(data, explanations) {
    var divs = ['bias', 'robustness', 'datasetQuality', 'explainability'];
    var names = ["biased", "Robustness", "Dataset Quality", "Explainability"];
    for (var i in explanations) {
        var new_data = createData(data, i);
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
        circle.setAttribute("stroke-dasharray", new_data[new_data.length - 1]['value'] + ", 100");
        var circleText = document.getElementById(i + "Text");
        circleText.innerHTML = new_data[new_data.length - 1]['value'];
    }

}


function redoMetrics() {
    var date1 = document.getElementById("startDate").value;
    var date2 = document.getElementById("endDate").value;
    return fetch('/getCertification/' + date1 + '/' + date2)
        .then(function (response) {
            return response.json();
        }).then(function (text) {
            redoMetrics2(text)
        });
}

function redoMetrics2(data) {
    for (var type in graphs) {
        var new_data = createData(data, type);
        graphs[type].setData(new_data);

        var circle = document.getElementById(type + "Circle");
        circle.setAttribute("stroke-dasharray", new_data[new_data.length - 1]['value'] + ", 100");
        var circleText = document.getElementById(type + "Text");
        circleText.innerHTML = new_data[new_data.length - 1]['value'];

    }
}

