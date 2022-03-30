from math import exp
import pandas as pd
import datetime
import json
import redis
from redis import StrictRedis
import subprocess
import threading
from RAI import utils
from RAI.metrics.registry import registry
from RAI.certificates import CertificateManager

class AISystem:
    def __init__(self, meta_database, dataset, task, user_config, custom_certificate_location=None) -> None:
        if type(user_config) is not dict:
            raise TypeError("User config must be of type Dictionary")
        self.meta_database = meta_database
        self.task = task
        self.dataset = dataset
        self.metric_groups = {}
        self.timestamp = ""
        self.sample_count = 0
        self.user_config = user_config
        self.certificate_manager = CertificateManager()
        if custom_certificate_location is None:
            self.certificate_manager.load_stock_certificates()
        else:
            self.certificate_manager.load_custom_certificates(custom_certificate_location)

    def initialize(self, metric_groups=None, metric_group_re=None, max_complexity="linear"):
        for metric_group_name in registry:
            metric_class = registry[metric_group_name]
            if metric_class.is_compatible(self):
                # if self._is_compatible(temp.compatibility):
                self.metric_groups[metric_group_name] = metric_class(self)
                print( f"metric group : {metric_group_name} was created" )

# May be more convenient to just accept metric name (or add functionality to detect group names and return a dictionary)
    def get_metric(self, metric_group_name, metric_name): 
        print(f"request for metric group : {metric_group_name}, metric_name : {metric_name}")
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
        raise Exception(f"unknown data type : {data_type}" )

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

    def compute_metrics(self, preds=None, reset_metrics=False, data_type="train", export_title=None):
        if reset_metrics:
            self.reset_metrics()
        data_dict = {"data": self.get_data(data_type)}
        if preds is not None:
            data_dict["predictions"] = preds
        for metric_group_name in self.metric_groups:
            self.metric_groups[metric_group_name].compute(data_dict)
        self.timestamp = self._get_time()
        self.sample_count += len(data_dict)
        if export_title is not None:
            self.export_data_flat(export_title)
            self.compute_certificates()
            self.export_certificates(description=export_title)

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
        cache = StrictRedis(host='localhost', port=6379, db=0)
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata", "certificate_values", "certificate"]
        cache.delete(*to_delete)
        return 

        r = redis.Redis(host='localhost', port=6379, db=0)
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata", "certificate_values", "certificate"]
        for key in to_delete:
            r.delete(self.task.model.name + "|" + key)


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


    def compute_certificates(self):
        metric_values = self.get_metric_values_flat()
        cert_values = self.certificate_manager.compute(metric_values)
        metadata = self.certificate_manager.metadata
        return cert_values, metadata


    def export_certificates(self, description=""):
        values = self.certificate_manager.results
        metadata = self.certificate_manager.metadata
        r = redis.Redis(host='localhost', port=6379, db=0)

        # metadata > date is added to metadata and values to allow for date based parsing of both and avoiding mismatch.
        values['metadata > date'] = {"value": self.metric_groups['metadata'].metrics['date'].value,
                                     "description": "time certificates were measured", "level": 1, "tags": ["metadata"]}
        values['metadata > description'] = {"value": description, "description": "Purpose of measurement.", "tags": ["metadata"]}
        metadata['metadata > date'] = {"value": self.metric_groups['metadata'].metrics['date'].value,
                                     "description": "time certificates were measured", "level": 1, "tags": ["metadata"]}
        metadata['metadata > description'] = {"value": "Measuring Stuff", "description": "Purpose of measurement.", "tags": ["metadata"]}


        r.set(self.task.model.name + '|certificate_metadata', json.dumps(metadata))
        r.rpush(self.task.model.name + '|certificate_values', json.dumps(values))  # True
        r.publish(self.task.model.name + "|certificate", values["metadata > date"]["value"])

    def get_certificate_values(self):
        result = {}
        for key in self.certificate_manager.results:
            if "metadata" not in self.certificate_manager.metadata[key]["tags"]:
                result[key] = self.certificate_manager.results[key]
        return result

    def get_certificate_category_summary(self):
        result = {"fairness": {"passed": [], "failed": [], "score": 0},
                  "robustness": {"passed": [], "failed": [], "score": 0},
                  "explainability": {"passed": [], "failed": [], "score": 0},
                  "performance": {"passed": [], "failed": [], "score": 0}}
        for key in self.certificate_manager.results:
            if "metadata" not in self.certificate_manager.metadata[key]["tags"]:
                if self.certificate_manager.results[key]["value"]:
                    result[self.certificate_manager.metadata[key]["tags"][0]]["passed"].append(key)
                else:
                    result[self.certificate_manager.metadata[key]["tags"][0]]["failed"].append(key)
        for key in result:
            result[key]["score"] = str(100*(len(result[key]["passed"]) / max(len(result[key]["failed"]), 1))) + "%"
        return result

    def get_certificate_category_scores(self):
        result = {}
        values = self.get_certificate_category_summary()
        for value in values:
            result[value] = values[value]["score"]
        return result

    def viewGUI(self):
        gui_launcher = threading.Thread(target=self._view_gui_thread, args=[])
        gui_launcher.start()

    def _view_gui_thread(self):
        subprocess.call("start /wait python GUI\\app.py " + self.task.model.name, shell=True)
        print("GUI can be viewed in new terminal")


    def set_agent(self, agent):
        self.task.model.agent = agent