import numpy as np


class DataSummarizer:
    def __init__(self, dataset, task, output_features):
        self.dataset = dataset 
        self.task = task
        self.train_data = self.dataset.data_dict["train"]
        self.test_data = self.dataset.data_dict["test"]
        self.output_features = output_features
        self.initialize()
        self.setLabelDistribution()

    def initialize(self):
        self.train_X = self.train_data.X
        self.train_y = self.train_data.y
        self.test_X = self.test_data.X
        self.test_y = self.test_data.y
        self.target = self.output_features[0].name
        self.y_name = self.output_features[0].values
        self.n_label = len(self.y_name)
        self.labels = [str(l) for l in range(self.n_label)]
        self.label_name_dict = None

        if self.task in ("binary_classification", "classification") and self.y_name is not None:
            self.label_name_dict = {l: self.y_name[int(l)] for l in self.labels}


    def setLabelDistribution(self):
        # dict with "train": distribution, "test": distribution
        train_y_dict, test_y_dict = dict(), dict()
        print(self.train_y[:10], self.labels)
        for label in self.labels:
            l_name = label
            if self.label_name_dict is not None:
                l_name = self.label_name_dict[label]
            train_y_dict[l_name] = int((self.train_y==int(label)).sum())
            test_y_dict[l_name] = int((self.test_y==int(label)).sum())
        self.label_dist_dict = {"train": train_y_dict, "test": test_y_dict}

    def getLabels(self):
        return self.labels 

    def getLabelDistribution(self):
        return self.label_dist_dict

    def getLabelNameDict(self):
        return self.label_name_dict

    def sampleDataPerLabel(self):
        pass
