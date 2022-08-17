import json
import subprocess
import threading
import redis
import RAI
import pickle
import os
import numpy as np 
from json import JSONEncoder
import logging
from RAI.Analysis import AnalysisManager
import codecs

logger = logging.getLogger(__name__)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


__all__ = ['RaiRedis']


class RaiRedis:
    """
    RaiRedis is used to provide Redis functionality. Allows for adding measurements, deleting measurements,
    exporting metadata. Standard port: 6379, db=0.
    """

    def __init__(self, ai_system: RAI.AISystem = None) -> None:
        self.redis_connection = None
        self.ai_system = ai_system
        self._ai_request_pub = None
        self._threads = []
        self.analysis_manager = AnalysisManager()

    def connect(self, host: str = "localhost", port: int = 6379) -> bool:
        self.redis_connection = redis.Redis(host=host, port=port, db=0)
        self._init_analysis_pubsub()
        return self.redis_connection.ping()

    def get_progress_update_lambda(self, analysis):
        return lambda progress: self.analysis_progress_update(analysis, progress)

    def analysis_progress_update(self, analysis: str, progress):
        self.redis_connection.publish("ai_requests", self.ai_system.name + '|start_analysis_update|' +
                                      analysis + "|" + progress)

    def _init_analysis_pubsub(self):
        def sub_handler(msg):
            logger.info(f"New Analysis message received: {msg}")
            msg = msg['data'].decode("utf-8").split('|')
            if msg[0] == self.ai_system.name:
                if msg[1] == "available_analysis":  # Request for the available analysis
                    available = self.analysis_manager.get_available_analysis(self.ai_system, msg[2])
                    self.redis_connection.publish("ai_requests", self.ai_system.name+'|available_analysis_response|'+json.dumps(available))
                elif msg[1] == "start_analysis":  # Request to start a specific analysis
                    dataset = msg[2]
                    analysis = msg[3]
                    if analysis in self.analysis_manager.get_available_analysis(self.ai_system, msg[2]):
                        connection = self.get_progress_update_lambda(analysis)
                        x = threading.Thread(target=self._run_analysis_thread, args=(dataset, analysis, connection))
                        x.start()

        self._ai_request_pub = self.redis_connection.pubsub()
        try:
            logger.info("channel subscribed")
            self._ai_request_pub.subscribe(**{"ai_requests": sub_handler})
            self._threads.append(self._ai_request_pub.run_in_thread(sleep_time=.1))
        except:
            logger.warning("unable to subscribe to redis pub/sub")

    def _run_analysis_thread(self, dataset, analysis, connection):
        result = self.analysis_manager.run_analysis(self.ai_system, dataset, analysis, connection=connection)
        encoded_res = codecs.encode(pickle.dumps(result[analysis].to_html()), "base64").decode()
        self.redis_connection.publish("ai_requests", self.ai_system.name + '|start_analysis_response|' +
                                      analysis + "|" + encoded_res)

    def reset_redis(self, export_metadata: bool = True, summarize_data: bool = False, interpret_model: bool = True) -> None:
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata",
                     "certificate_values", "certificate"]
        for key in to_delete:
            self.redis_connection.delete(self.ai_system.name + "|" + key)
        if export_metadata:
            self.export_metadata()
        if summarize_data:
            self.summarize()
        if interpret_model:
            self.interpret_model()
        self.redis_connection.publish('update', "cleared")

    def delete_data(self, system_name) -> None:
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata",
                     "certificate_values", "certificate", "certificate_info", "project_info"]
        for key in to_delete:
            self.redis_connection.delete(system_name + "|" + key)
        self.redis_connection.srem("projects", system_name)
        self.redis_connection.publish('update', "cleared")

    def delete_all_data(self, confirm=False):
        if confirm:
            print("Deleting!")
            for key in self.redis_connection.scan_iter("*|project_info"):
                val = key[:-13].decode("utf-8")
                print("Deleting: ", val)
                self.delete_data(val)

    def export_metadata(self) -> None:
        metric_info = self.ai_system.get_metric_info()
        certificate_info = self.ai_system.get_certificate_info()
        project_info = self.ai_system.get_project_info()

        self.redis_connection.set(self.ai_system.name + '|metric_info', json.dumps(metric_info))
        self.redis_connection.set(self.ai_system.name + '|certificate_info', json.dumps(certificate_info))
        self.redis_connection.set(self.ai_system.name + '|project_info', json.dumps(project_info))
        self.redis_connection.sadd("projects", self.ai_system.name)

    def export_visualizations(self) -> None:
        print("AI System Name: ", self.ai_system.name)
        self.summarize()
        self.interpret_model()

    def summarize(self):
        data_summary = self.ai_system.get_data_summary()
        self.redis_connection.set(self.ai_system.name + '|data_summary', json.dumps(data_summary))

    def add_measurement(self) -> None:
        certificates = self.ai_system.get_certificate_values()
        metrics = self.ai_system.get_metric_values()
        print("Sharing: ", self.ai_system.name)
        self.redis_connection.rpush(self.ai_system.name + '|certificate_values', json.dumps(certificates))  # True
        # Leaving this for now.
        # TODO: Set up standardized to json for all metrics.
        '''
        # print("METRICS: ", metrics)
        for dataset in metrics:
            for group in metrics[dataset]:
                for m in metrics[dataset][group]:
                    print(m, "\n")
                    
        print("testing json dumps: \n")
        for dataset in metrics:
            for group in metrics[dataset]:
                for m in metrics[dataset][group]:
                    if "moment" in m:
                        continue
                    print(m, "\n")
                    print(metrics[dataset][group][m])
                    print(json.dumps(metrics[dataset][group][m]))
        '''

        self.redis_connection.rpush(self.ai_system.name + '|metric_values', json.dumps(metrics))  # True
        self.redis_connection.publish('update',
                                      "New measurement: %s" % metrics[list(metrics.keys())[0]]["metadata"]["date"])

    def interpret_model(self):
        interpretation = self.ai_system.get_interpretation()
        self.redis_connection.set(self.ai_system.name + '|model_interpretation', json.dumps(interpretation, cls=NumpyArrayEncoder))

    def viewGUI(self):
        gui_launcher = threading.Thread(target=self._view_gui_thread, args=[])
        gui_launcher.start()

    def _view_gui_thread(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("../../Dashboard")
        file = os.path.abspath("main.py")
        subprocess.call("start /wait python " + file, shell=True)
        print("GUI can be viewed in new terminal")
