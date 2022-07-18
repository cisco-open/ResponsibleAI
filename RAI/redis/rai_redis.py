import json
import subprocess
import threading
import redis
import RAI

__all__ = ['RaiRedis']


class RaiRedis:
    """
    RaiRedis is used to provide Redis functionality. Allows for adding measurements, deleting measurements,
    exporting metadata. Standard port: 6379, db=0.
    """

    def __init__(self, ai_system: RAI.AISystem = None) -> None:
        self.redis_connection = None
        self.ai_system = ai_system

    def connect(self, host: str = "localhost", port: int = 6379) -> bool:
        self.redis_connection = redis.Redis(host=host, port=port, db=0)
        return self.redis_connection.ping()

    def reset_redis(self, export_metadata: bool = True, summarize_data: bool = True) -> None:
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata",
                     "certificate_values", "certificate"]
        for key in to_delete:
            self.redis_connection.delete(self.ai_system.name + "|" + key)
        if export_metadata:
            self.export_metadata()
        if summarize_data:
            self.summarize_data()
        self.redis_connection.publish('update', "cleared")

    def delete_data(self, system_name) -> None:
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata",
                     "certificate_values", "certificate", "certificate_info", "project_info"]
        for key in to_delete:
            self.redis_connection.delete(system_name + "|" + key)
        self.redis_connection.srem("projects", system_name)
        self.redis_connection.publish('update', "cleared")

    def export_metadata(self) -> None:
        metric_info = self.ai_system.get_metric_info()
        certificate_info = self.ai_system.get_certificate_info()
        project_info = self.ai_system.get_project_info()

        self.redis_connection.set(self.ai_system.name + '|metric_info', json.dumps(metric_info))
        self.redis_connection.set(self.ai_system.name + '|certificate_info', json.dumps(certificate_info))
        self.redis_connection.set(self.ai_system.name + '|project_info', json.dumps(project_info))
        self.redis_connection.sadd("projects", self.ai_system.name)

    def summarize_data(self) -> None:
        print("AI System Name: ", self.ai_system.name)
        data_summary = self.ai_system.get_data_summary()
        print("Data Summary: ", data_summary)
        self.redis_connection.set(self.ai_system.name + '|data_summary', json.dumps(data_summary))

    def add_measurement(self) -> None:
        certificates = self.ai_system.get_certificate_values()
        metrics = self.ai_system.get_metric_values()

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

    def viewGUI(self):
        gui_launcher = threading.Thread(target=self._view_gui_thread, args=[])
        gui_launcher.start()

    def _view_gui_thread(self):
        subprocess.call("start /wait python Dashboard\\main.py ", shell=True)
        print("GUI can be viewed in new terminal")
