__all__ = ['RaiRedis']

import json
import redis
import RAI
import subprocess
import threading
from RAI import utils

class RaiRedis:
    def __init__(self, ai_system:RAI.AISystem = None) -> None:
        self.ai_system = ai_system
        self.auto_id = 0 



    def connect(self, host:str = "localhost", port:int = 6379) -> bool:
        self.redis_connection = redis.Redis(host='localhost', port=6379, db=0)
        return self.redis_connection.ping()



    def reset_redis(self, export_metadata:bool = True) -> None :
         
        to_delete = ["metric_values", "model_info", "metric_info", "metric", "certificate_metadata", "certificate_values", "certificate"]
        
        for key in to_delete:
            self.redis_connection.delete(self.ai_system.task.model.name + "|" + key)

        if export_metadata:
            self.export_metadata()
            
    def export_metadata(self) -> None :
        metric_info = self.ai_system.get_metric_info()
        certificate_info = self.ai_system.get_certificate_info()
        model_info = self.ai_system.get_model_info()
        
        self.redis_connection.set(self.ai_system.task.model.name + '|metric_info', json.dumps(metric_info))
        self.redis_connection.set(self.ai_system.task.model.name + '|certificate_info', json.dumps(certificate_info))
        self.redis_connection.set(self.ai_system.task.model.name + '|model_info', json.dumps(model_info))


    def add_measurement(self, tag:str = None) -> None :

        if tag is None:
            tag = str(self.auto_id)
            self.auto_id+=1


        certificates = self.ai_system.get_certificate_values()
        metrics = self.ai_system.get_metric_values()


        # certificates['metadata > date'] = {"value": self.ai_system._timestamp,
        #                              "description": "time certificates were measured", "level": 1, "tags": ["metadata"]}
        # certificates['metadata > description'] = {"value": tag, "description": "Purpose of measurement.", "tags": ["metadata"]}
        self.redis_connection.rpush(self.ai_system.task.model.name + '|certificate_values', json.dumps(certificates))  # True

        
        # metrics['metadata > description'] = tag
        self.redis_connection.rpush(self.ai_system.task.model.name + '|metric_values', json.dumps(metrics))  # True

    def viewGUI(self):
        gui_launcher = threading.Thread(target=self._view_gui_thread, args=[])
        gui_launcher.start()

    def _view_gui_thread(self):
        subprocess.call("start /wait python GUI\\app.py " + self.ai_system.task.model.name, shell=True)
        print("GUI can be viewed in new terminal")
