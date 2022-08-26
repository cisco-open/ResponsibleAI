import json
import logging
from collections import defaultdict
import numpy as np
import redis
import dash_bootstrap_components as dbc
from dash import html
logger = logging.getLogger(__name__)


class RedisUtils(object):
    def __init__(self, host="localhost", port=6379, db=0, precision=3, text_maxlen=100):
        self._host = host
        self._port = port
        self._db = db
        self._threads = []
        self._precision = precision
        self._maxlen = text_maxlen
        self.values = {}
        self.info = {}
        self._initialized = False
        self._subscribers = defaultdict(bool)
        self._current_project = {}  # Contains certificates, metrics, info, project info
        self._current_project_name = None  # Used with redis to get current project
        self._ai_request_pub = None
        self._load_analysis_storage = {}

    def has_update(self, channel, reset=True):
        val = self._subscribers[channel]
        if reset:
            self.reset_channel(channel)
        return val

    def has_analysis_update(self, analysis, reset=True):
        channel = "analysis_update|" + self._current_project_name + "|" + analysis
        val = self._subscribers.get(channel, False)
        if reset:
            self.reset_channel(channel)
        return val

    def reset_channel(self, channel):
        self._subscribers[channel] = False

    def _init_pubsub(self):
        def sub_handler(msg):
            logger.info(f"a new message received: {msg}")
            self._update_projects()
            if self._current_project_name:
                self._update_info()
                self._update_values()
            relevant = ["metric_detail", "metric_graph", "certificate"]
            for item in relevant:
                self._subscribers[item] = True
        self._redis_pub = self._redis.pubsub()

        try:
            logger.info("channel subscribed")
            self._redis_pub.subscribe(**{"update": sub_handler})
            self._threads.append(self._redis_pub.run_in_thread(sleep_time=.1))
        except:
            logger.warning("unable to subscribe to redis pub/sub")

    def _init_analysis_pubsub(self):
        def sub_handler(msg):
            msg = msg['data'].decode("utf-8")
            msg_split = msg.split('|')
            project_name = msg_split[0]
            if msg_split[1] == "available_analysis_response":
                if project_name == self._current_project_name:
                    val = json.loads(msg_split[2])
                    self.set_available_analysis(val)
                    self._subscribers["analysis_update"] = True
                    logger.info("Setting available analysis to: " + str(self.get_available_analysis()))
            elif msg_split[1] == "start_analysis_response":  # update ping or final analysis
                analysis_name = msg_split[2]
                print("Analysis received")
                report = self._redis.get(self._current_project_name+"|analysis|"+analysis_name)
                self.set_analysis_progress(project_name, analysis_name, html.Div(json.loads(report)))
                # self.set_analysis_progress(project_name, analysis_name, None)
                self._subscribers["analysis_update|" + project_name + "|" + analysis_name] = True
            elif msg_split[1] == "start_analysis_update":  # update ping or final analysis
                analysis_name = msg_split[2]
                progress = msg_split[3]
                print("Analysis update recieved")
                self.set_analysis_progress(project_name, analysis_name, self._progress_to_html(progress))
                self._subscribers["analysis_update|" + project_name + "|" + analysis_name] = True

        self._ai_request_pub = self._redis.pubsub()
        try:
            self._ai_request_pub.subscribe(**{"ai_requests": sub_handler})
            self._threads.append(self._ai_request_pub.run_in_thread(sleep_time=.1))
            logger.info("channel subscribed ai_requests")
        except:
            logger.warning("unable to subscribe to ai_requests pub/sub")

    def _progress_to_html(self, progress):
        return html.Div(dbc.Progress(value=progress, label=str(progress) + "%"))

    def request_start_analysis(self, analysis):
        self.set_analysis_progress(self._current_project_name, analysis, html.Div())
        self._redis.publish('ai_requests', self._current_project_name + "|start_analysis|" +
                            self.get_current_dataset() + "|" + analysis)

    def request_available_analysis(self):
        print("requesting available analysis")
        self._redis.publish('ai_requests', self._current_project_name + "|available_analysis|" + self.get_current_dataset())

    def initialize(self, subscribers=None):
        if self._initialized:
            return
        self._redis = redis.Redis(self._host, self._port, self._db)

        self._subscribers = {}
        if subscribers is not None:
            for item in subscribers:
                self._subscribers[item] = False

        self._init_pubsub()
        self._update_projects()
        self._init_analysis_pubsub()
        self._initialized = True

    def reformat(self, precision):
        self._precision = precision
        self._current_project = self._reformat_data(self._current_project)

    def _reload(self):
        self._update_projects()
        self._update_info()
        self._update_values()

    def close(self):
        if self._redis_pub is not None:
            try:
                for t in self._threads:
                    t.stop()
                self._redis_pub.unsubscribe(self._model_name + "|update")
                self._redis_pub.close()
            except:
                logger.warning("Problem in closing pub/sub")
        self._redis.close()

    def get_project_info(self):
        return self._current_project.get("project_info", {})

    def get_metric_info(self):
        return self._current_project.get("metric_info", {})

    def get_certificate_info(self):
        return self._current_project.get("certificate_info", {})

    def get_certificate_values(self):
        return self._current_project.get("certificate_values", {})

    def get_metric_values(self):
        return self._current_project.get("metric_values", {})

    def get_current_dataset(self):
        return self._current_project.get("current_dataset", None)

    def get_data_summary(self):
        return self._current_project.get("data_summary", {})

    def get_model_interpretation(self):
        return self._current_project.get("model_interpretation", {})

    def get_available_analysis(self):
        return self._current_project.get("available_analysis", [])

    def get_analysis(self, analysis_name):
        # return self._load_analysis_storage.get(self._current_project_name, {}).get(analysis_name, None)
        analysis = None
        if analysis_name is not None:
            analysis = self._load_analysis_storage.get(self._current_project_name, {}).get(analysis_name, None)
            if analysis is None:
                analysis = self._redis.get(self._current_project_name+"|analysis|"+analysis_name)
                if analysis is not None:
                    analysis = json.loads(analysis)
        return analysis

    def set_current_project(self, project_name):
        project_name = project_name
        logger.info(f"changing current project from {self._current_project_name} to {project_name}")
        if self._current_project_name == project_name:
            return
        self._current_project_name = project_name
        self._current_project = {}
        self._update_info()
        self._update_values()
        self.set_data_summary()
        self.set_model_interpretation()
        available_analysis = self._query_existing_analysis()
        self.set_available_analysis(available_analysis)
        self._current_project["analysis"] = {}
        self._current_project = self._reformat_data(self._current_project)

    def set_current_dataset(self, dataset):
        self._current_project["current_dataset"] = dataset

    def set_data_summary(self):
        print("Current proj name: ", self._current_project_name)
        summary = self._redis.get(self._current_project_name + "|data_summary")
        self._current_project["data_summary"] = json.loads(summary) if summary is not None else {}

    def set_model_interpretation(self):
        interpretation = self._redis.get(self._current_project_name + "|model_interpretation")
        self._current_project["model_interpretation"] = json.loads(interpretation) if interpretation is not None else {}

    def set_available_analysis(self, available):
        self._current_project["available_analysis"] = available

    def _query_existing_analysis(self):
        available = []
        if self._current_project_name is not None:
            for key in self._redis.scan_iter(self._current_project_name+"|analysis|*"):
                key = key.decode("utf-8")
                available.append(key[len(self._current_project_name + "|analysis|"):])
            return available

    def set_analysis_progress(self, project_name, analysis_name, report):
        if project_name not in self._load_analysis_storage:
            self._load_analysis_storage[project_name] = {}
        self._load_analysis_storage[project_name][analysis_name] = report

    def _update_projects(self):
        self._projects = self._redis.smembers("projects")
        self._projects = [s.decode('utf-8') for s in self._projects]

    def get_projects_list(self):
        return self._projects

    def get_dataset_list(self):
        return self._current_project.get("dataset_values", [])

    def _update_info(self):
        self.info = {}
        print("current project name: ", self._current_project_name)
        self._current_project["project_info"] = \
            json.loads(self._redis.get(self._current_project_name + '|project_info'))
        self._current_project["certificate_info"] = \
            json.loads(self._redis.get(self._current_project_name + '|certificate_info'))
        self._current_project["metric_info"] = \
            json.loads(self._redis.get(self._current_project_name + '|metric_info'))

    def _update_values(self):
        self.values = {}
        self._current_project["metric_values"] = []
        for data in self._redis.lrange(self._current_project_name + '|metric_values', 0, -1):
            self._current_project["metric_values"].append(json.loads(data))

        self._current_project["certificate_values"] = []
        for data in self._redis.lrange(self._current_project_name + '|certificate_values', 0, -1):
            self._current_project["certificate_values"].append(json.loads(data))

        self._current_project["dataset_values"] = []
        for item in self._current_project["metric_values"]:
            for val in item:
                if val not in self._current_project["dataset_values"]:
                    self._current_project["dataset_values"].append(val)

    def _reformat_data(self, x):
        if type(x) is float:
            return np.round(x, self._precision)
        elif type(x) is list:
            return [self._reformat_data(i) for i in x]
        elif isinstance(x, dict):
            return {k: self._reformat_data(v) for k, v in x.items()}
        else:
            return x

    def __del__(self):
        pass
    # self.close()
