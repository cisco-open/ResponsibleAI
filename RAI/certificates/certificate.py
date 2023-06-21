# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import json
from RAI import certificates
from RAI import utils

__all__ = ['Certificate']


class Certificate(object):
    """
    Certificate Objects contain information about a particular certificate.
    Certificates are automatically loaded in by CertificateManagers and perform evaluation using metric
    data they are provided in combination with the certificate data loaded in.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cert_spec = {}
        self.term_values = []

    # Loads certificate information from a json file
    def load_from_json(self, json_file):
        """
        opens the certificate json file and load all the information

        :param json_file: json_file file path of the certificate json file is shared

        :return: None
        """
        f = open(json_file, "r")
        self.cert_spec = json.load(f)

    def evaluate(self, metrics, certs):
        """
        From the certificate json file condition key is selected and based on that evalutions will happen


        :param metrics: metrics data is provided based on that evaluation will happen
        :param certs: certificate data is provided based on that evaluation will happen

        :return:  Returns the evaluation result based on the input data
        """
        cond = self.cert_spec["condition"]
        return self._do_eval(cond, metrics, certificates)

    def _do_eval(self, cond, metrics, certs):
        op = cond["op"]
        terms = cond["terms"]
        self.term_values = []
        for t in terms:
            if isinstance(t, dict):
                self.term_values.append(self._do_eval(t, metrics, certs))
            else:
                self.term_values.append(self._eval_atomic(t, metrics, certs))
        if op == "and":
            return all(self.term_values)
        if op == "or":
            return any(self.term_values)
        raise Exception("unknown data type : {}".format(cond))

    def _eval_atomic(self, t, metrics, certs):
        var, op, value = t
        if value == "True":
            value = 1
        if value == "False":
            value = 0
        value = float(value)
        v = self._get_value(var, metrics, certs)
        result = False
        if op == "==" or op == "=":
            result = v == value
        if op == ">":
            result = v > value
        if op == ">=":
            result = v >= value
        if op == "<":
            result = v < value
        if op == "<=":
            result = v <= value
        return utils.jsonify(result)

    def _get_value(self, var, metrics, certs):
        if var[0] == "&":
            group, metric = var[1:].split(" > ")
            return metrics[group][metric]
        if var[0] == "@":
            return certs[var[1:]]
