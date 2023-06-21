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


class DataSummarizer:
    def __init__(self, dataset, task, output_features):
        self.dataset = dataset
        self.label_name_dict = {str(val): output_features[0].values[val] for val in output_features[0].values} \
            if output_features[0].values is not None else {}
        self.task = task
        self.train_data = self.dataset.data_dict.get("train", None)
        self.test_data = self.dataset.data_dict.get("train", None)
        self.output_features = output_features
        self.initialize()
        if self.test_data is not None and self.train_data is not None:
            self.setLabelDistribution()

    def initialize(self):
        self.train_X = self.train_data.X if self.train_data is not None else None
        self.train_y = self.train_data.y if self.train_data is not None else None
        self.test_X = self.test_data.X if self.train_data is not None else None
        self.test_y = self.test_data.y if self.train_data is not None else None
        self.target = self.output_features[0].name
        self.y_name = None
        self.n_label = None
        self.labels = None
        self.label_name_dict = None
        self.label_dist_dict = None

        if self.task in ("binary_classification", "classification") and self.y_name is not None:
            self.label_name_dict = {lab: self.y_name[int(lab)] for lab in self.labels}
            self.y_name = self.output_features[0].values
            self.n_label = len(self.y_name)
            self.labels = [str(nl) for nl in range(self.n_label)]

    def setLabelDistribution(self):
        # dict with "train": distribution, "test": distribution
        train_y_dict, test_y_dict = dict(), dict()

        print(self.train_y[:10], self.labels)
        if self.labels is not None:
            for label in self.labels:
                l_name = label
                if self.label_name_dict is not None:
                    l_name = self.label_name_dict[label]
                train_y_dict[l_name] = int((self.train_y == int(label)).sum())
                test_y_dict[l_name] = int((self.test_y == int(label)).sum())
            self.label_dist_dict = {"train": train_y_dict, "test": test_y_dict}

    def getLabels(self):
        return self.labels

    def getLabelDistribution(self):
        return self.label_dist_dict

    def getLabelNameDict(self):
        return self.label_name_dict

    def sampleDataPerLabel(self):
        pass
