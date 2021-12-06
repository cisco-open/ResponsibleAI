import pandas as pd
import datetime
from RAI.metrics.registry import registry
import json
import redis
import subprocess
import random
from RAI import utils
import threading
from RAI.certificates import CertificateManager


class AISystem:
    def __init__(self, meta_database, dataset, task, user_config) -> None:
        self.meta_database = meta_database
        self.task = task
        self.dataset = dataset
        self.metric_groups = {}
        self.timestamp = ""
        self.sample_count = 0
        self.user_config = user_config

        self.certificate_manager = CertificateManager()
        self.certificate_manager.load_stock_certificates()
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
                  "task_type": self.task.type, "configuration": self.user_config, "features": [], "description": self.task.description,
                  "display_name": self.task.model.display_name}
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
        r.publish(self.task.model.name + "|metric", metric_values['metadata > date'])
        # r.save()

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
        # CERTIFICATE VALUES, one for each group (Fairness, Explainability etc.). I added one for each level.

        metric_values = self.get_metric_values_flat()
        cert_values = self.certificate_manager.compute( metric_values)
        # cert_values = { "cert1_1": {"value": True, "explanation": "Test Passed because ..."},
        #                 "cert1_2": {"value": True, "explanation": "Test Passed because .."},
        #                 "cert2_1": {"value": False, "explanation": "Test failed because .."},
        #                 "cert2_2": {"value": True, "explanation": "Test passed because .."},
        #                 "cert3_1": {"value": True, "explanation": "Test passed because .."},
        #                 "cert3_2": {"value": False, "explanation": "Test failed because .."},
        #                 "cert4_1": {"value": True, "explanation": "Test passed because .."},
        #                 "cert4_2": {"value": True, "explanation": "Test passed becuase ..."}}


        # it is currently important that there is at least one metric for each metric group for displaying data on the main page.
        # Relevant information on each metric
        metadata = self.certificate_manager.metadata

        # metadata = {    "cert1_1": {"display_name": "Cert 1 Low Level", "tags": ["fairness"], "level": 1, "description": "A Level 1 Fairness Certificate"},
        #                 "cert1_2": {"display_name": "Cert 1 High Level", "tags": ["fairness"], "level": 2, "description": "A Level 2 Fairness Certificate"},
        #                 "cert2_1": {"display_name": "Cert 2 Low Level", "tags": ["robust"], "level": 1, "description": "A Level 1 Robustness Certificate"},
        #                 "cert2_2": {"display_name": "Cert 2 High Level", "tags": ["robust"], "level": 2, "description": "A Level 2 Robustness Certificate"},
        #                 "cert3_1": {"display_name": "Cert 1 Low Level", "tags": ["explainability"], "level": 1, "description": "A Level 1 Explainability Certificate"},
        #                 "cert3_2": {"display_name": "Cert 1 High Level", "tags": ["explainability"], "level": 2, "description": "A Level 2 Explainability Certificate"},
        #                 "cert4_1": {"display_name": "Cert 1 Low Level", "tags": ["performance"], "level": 1, "description": "A Level 1 Performance Certificate"},
        #                 "cert4_2": {"display_name": "Cert 1 High Level", "tags": ["performance"], "level": 2, "description": "A Level 2 Performance Certificate"}}

        return cert_values, metadata

# TEMPORARY FUNCTION FOR PRODUCING DATA
    def export_certificates(self):
        values, metadata = self.compute_certificates()
        r = redis.Redis(host='localhost', port=6379, db=0)


        # metadata > date is added to metadata and values to allow for date based parsing of both and avoiding mismatch.
        values['metadata > date'] = {"value": self.metric_groups['metadata'].metrics['date'].value,
                                     "description": "time certificates were measured", "level": 1, "tags": ["metadata"]}
        values['metadata > description'] = {"value": "Measuring Stuff", "description": "Purpose of measurement.", "tags": ["metadata"]}

        metadata['metadata > date'] = {"value": self.metric_groups['metadata'].metrics['date'].value,
                                     "description": "time certificates were measured", "level": 1, "tags": ["metadata"]}
        metadata['metadata > description'] = {"value": "Measuring Stuff", "description": "Purpose of measurement.", "tags": ["metadata"]}


        r.set(self.task.model.name + '|certificate_metadata', json.dumps(metadata))
        r.rpush(self.task.model.name + '|certificate_values', json.dumps(values))  # True
        r.publish(self.task.model.name + "|certificate", values["metadata > date"]["value"])

    def viewGUI(self):
        gui_launcher = threading.Thread(target=self._view_gui_thread, args=[])
        gui_launcher.start()

    def _view_gui_thread(self):
        subprocess.call("start /wait python GUI\\app.py " + self.task.model.name, shell=True)
        print("GUI can be viewed in new terminal")

