import numpy as np

class DataSummarizer:
    def __init__(self, dataset):
        self.dataset = dataset 
        self.train_data = self.dataset.data_dict["train"]
        self.test_data = self.dataset.data_dict["test"]
        self.initialize()
        self.setLabelDistribution()

    def initialize(self):
        self.train_X = self.train_data.X
        self.train_y = self.train_data.y
        self.test_X = self.test_data.X
        self.test_y = self.test_data.y
        self.labels = np.unique(np.concatenate([self.train_y, self.test_y], axis=0)).tolist()
        self.n_label = len(self.labels)

    def setLabelDistribution(self):
        # dict with "train": distribution, "test": distribution
        train_y_dict, test_y_dict = dict(), dict()
        for label in self.labels:
            train_y_dict[label] = int((self.train_y==label).sum())
            test_y_dict[label] = int((self.test_y==label).sum())
        self.label_dist_dict = {"train": train_y_dict, "test": test_y_dict}

    def getLabels(self):
        return self.labels 

    def getLabelDistribution(self):
        return self.label_dist_dict

    def sampleDataPerLabel(self):
        pass
