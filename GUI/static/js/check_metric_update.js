// Code for the event list page
'use strict';


$(document).ready(function() {
        setInterval("check_data()", 2000); // call every 10 seconds
});

function check_data() {
   fetch('/updateMetrics').then(function (response) {
        return response.json();
    }).then(function(result){
        if (result){
            location.reload()
        }
    });
}

