import numpy as np
import pandas as pd
import sklearn as sk
import datetime
from RAI.metrics.registry import registry
import json
import redis
import subprocess
import random
from RAI import utils


class AISystem:
    def __init__(self, meta_database, dataset, task, user_config) -> None:
        self.meta_database = meta_database
        self.task = task
        self.dataset = dataset
        self.metric_groups = {}
        self.timestamp = ""
        self.sample_count = 0
        self.user_config = user_config

    def initialize(self, metric_groups=None, metric_group_re=None, max_complexity="linear"):
        for metric_group_name in registry:
            metric_class = registry[metric_group_name]
            if metric_class.is_compatible(self):
                # if self._is_compatible(temp.compatibility):
                self.metric_groups[metric_group_name] = metric_class(self)
                print("metric group : {} was created".format(metric_group_name))

# May be more convenient to just accept metric name (or add functionality to detect group names and return a dictionary)
    def get_metric(self, metric_group_name, metric_name): 
        print("request for metric group : {}, metric_name : {}".format(metric_group_name, metric_name))
        return self.metric_groups[metric_group_name].metrics[metric_name].value

    def reset_metrics(self):
        for metric_group_name in self.metric_groups:
           self.metric_groups[metric_group_name].reset()
        self.sample_count = 0
        self.time_stamp = None  # Replace by registering a time metric in metric_groups? 

    def get_data(self, data_type):
        if data_type == "train":
            return self.dataset.train_data
        if data_type == "val":
            return self.dataset.val_data
        if data_type == "test":
            return self.dataset.test_data
        raise Exception("unknown data type : {}".format(data_type))

    def get_model_info(self):
        result = {"id": self.task.model.name, "model": self.task.model.model_class, "adaptive": self.task.model.adaptive,
                  "task_type": self.task.type, "configuration": self.user_config, "features": [], "description": self.task.description}
        for i in range(len(self.meta_database.features)):
            result['features'].append(self.meta_database.features[i].name)
        return result

    def get_metric_info_flat(self):
        result = {}
        for group in self.metric_groups:
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[metric_obj.unique_name] = metric_obj.config
                metric_obj.config["tags"] = self.metric_groups[group].tags  # Change this up after
        return result

    def get_metric_info_dict(self):
        result = {}
        for group in self.metric_groups:
            result[ group ] = {}
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[group][metric] = metric_obj.config
                 
        return result


    def get_metric_values_flat(self):
        result = {}
        for group in self.metric_groups:
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[metric_obj.unique_name] = utils.jsonify(metric_obj.value)
               
        return result
    def get_metric_values_dict(self):
        result = {}
        for group in self.metric_groups:
            result[ group ] = {}
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                result[group][metric] = utils.jsonify(metric_obj.value)
                 
        return result

    def compute_metrics(self, preds=None, reset_metrics=False, data_type="train"):
        if reset_metrics:
            self.reset_metrics()
        data_dict = {"data": self.get_data(data_type)}
        if preds is not None:
            data_dict["predictions"] = preds
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].compute(data_dict)
        self.timestamp = self._get_time()
        self.sample_count += len(data_dict)

    def update_metrics(self, data):
        for i in range(len(data)):
            for metric_group_name in self.metric_groups:
                self.metric_groups[metric_group_name].update(data[i])
        self.timestamp = self._get_time()
        self.sample_count += 1

    def export_metric_values(self):
        result = {}
        for metric_group_name in self.metric_groups:
            result[metric_group_name] = self.metric_groups[metric_group_name].export_metric_values()
        return result

    def _get_time(self):
        now = datetime.datetime.now()
        return "{:02d}".format(now.year) + "-" + "{:02d}".format(now.month) + "-" + "{:02d}".format(now.day) + " " + "{:02d}".format(now.hour) + ":" + "{:02d}".format(now.minute) + ":" + "{:02d}".format(now.second)

    def export_data_flat(self, description=""):
        metric_values = self.get_metric_values_flat()
        metric_info = self.get_metric_info_flat()
        model_info = self.get_model_info()
        metric_values['metadata > description'] = description
        self._update_redis(metric_values, model_info, metric_info)

    def export_data_dict(self):
        metric_values = self.get_metric_values_dict()
        metric_values["date"] = self._get_time()  # temporary solution
        metric_info = self.get_metric_info_dict()
        model_info = self.get_model_info()
        self._update_redis(metric_values, model_info, metric_info)


    def reset_redis(self):
        r = redis.Redis(host='localhost', port=6379, db=0)
        for key in r.keys():
            r.delete(key)
            
    def _dict_to_csv(self, file, dict, write_headers=True):
        newDict = {}
        newDict['date'] = self.timestamp
        for category in dict:
            for metric in dict[category]:
                newDict[metric] = dict[category][metric]
        df = pd.DataFrame([newDict])
        df.to_csv(file, header=write_headers, mode='a', index=False)

    def _update_redis(self, metric_values, model_info, metric_info):
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.rpush(self.task.model.name + '|metric_values', json.dumps(metric_values))  # True
        r.set(self.task.model.name + '|model_info', json.dumps(model_info))
        r.set(self.task.model.name + '|metric_info', json.dumps(metric_info))
        r.save()

    # Searches all metrics. Queries based on Metric Name, Metric Group Name, Category, and Tags.
    def search(self, query):
        query = query.lower()
        results = {}
        for group in self.metric_groups:
            add_group = group.lower() == query 
            for metric in self.metric_groups[group].metrics:
                metric_obj = self.metric_groups[group].metrics[metric]
                if add_group or metric.lower().find(query) > -1 or metric_obj.display_name.lower().find(query) > -1:
                    results[metric] = metric_obj.value
                elif metric_obj.tags is not None:
                    for tag in metric_obj.tags:
                        if tag.lower().find(query) > -1:
                            results[metric] = metric_obj.value
                            break
        return results


# TEMPORARY FUNCTION FOR PRODUCING DATA
    def compute_certificates(self):
        result = {}
        for group in self.metric_groups:
            if self.metric_groups[group].tags[0] not in result:
                result[self.metric_groups[group].tags[0]] = {}
                result[self.metric_groups[group].tags[0]]["list"] = {}
                result[self.metric_groups[group].tags[0]]["score"] = 0
            for metric in self.metric_groups[group].metrics:
                passed = bool(random.getrandbits(1))
                explanation = "filler"
                if passed:
                    result[self.metric_groups[group].tags[0]]["score"] += 1
                result[self.metric_groups[group].tags[0]]["list"][self.metric_groups[group].metrics[metric].unique_name] \
                    = {"score": passed, "explanation": explanation, "level": random.randint(1, 2)}
        return result

# TEMPORARY FUNCTION FOR PRODUCING DATA
    def export_certificates(self):
        values = self.compute_certificates()
        r = redis.Redis(host='localhost', port=6379, db=0)
        certificate_metadata = {"fairness": {"name": "fairness", "explanation": "Measures how fair a model's predictions are.", "display_name": "Fairness"}, "robust": {"name": "robustness", "explanation": "Measures a model's resiliance to time and sway.", "display_name": "Robustness"}, "explainability": {"name": "explainability", "explanation": "Measures how explainable the model is.", "display_name": "Explainability"}, "performance": {"name": "performance", "explanation": "Performance describes how well at predicting the model was.", "display_name": "Performance"}}

        values["explainability"] = {"score": 1, "list": {"temp function:": {"score": True, "explanation": "filler function"}}}
        values['metadata'] = {'date': self.metric_groups['metadata'].metrics['date'].value}

        r.set(self.task.model.name + '|certificate_metadata', json.dumps(certificate_metadata))
        r.rpush(self.task.model.name + '|certificate_values', json.dumps(values))  # True



    # def summarize(self):
    #     categories = {}
    #     # Separate Metric Groups by Category
    #     for group in self.metric_groups:
    #         if self.metric_groups[group].category.lower() not in categories:
    #             categories[self.metric_groups[group].category.lower()] = []
    #         categories[self.metric_groups[group].category.lower()].append(group)

    #     for category in categories:
    #         print("Category ", category, " Metrics")
    #         for group in categories[category]:
    #             print("\tGroup ", group)
    #             metric_values = self.metric_groups[group].get_metric_values()
    #             for metric in metric_values:
    #                 print("\t\t", metric, " ", metric_values[metric])


    def viewGUI(self):
        subprocess.call("start /wait python GUI\\app.py " + self.task.model.name, shell=True)
        print("GUI can be viewed in new terminal")
