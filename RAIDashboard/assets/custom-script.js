// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

function inputsChecked(inputs){
    for (var i = 0; i < inputs.length; i++) {
       if (inputs[i].checked == true){
            return true
       }
    }
    return false
}

function setButtonState(inputs, button){
    var isChecked = inputsChecked(inputs);
    if (button) {
        if (isChecked){
            button.style.visibility = 'visible';
        }
        else {
            button.style.visibility = 'hidden';
        }
    }
}


function groupedMetricsGraph() {
    var inputs = document.getElementsByClassName("grouped-checklist-input");
    var button = document.getElementsByClassName("grouped_reset_graph_div")[0];
    setButtonState(inputs, button)

}

function singleMetricGraph() {
    var inputs = document.getElementsByClassName("single-radio-input");
    var button = document.getElementsByClassName("single_reset_graph_div")[0];
    setButtonState(inputs, button)
}

setInterval(function() {
    groupedMetricsGraph()
    singleMetricGraph()


}, 300);