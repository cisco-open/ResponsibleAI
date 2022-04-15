import os
import json
import redis
import sys
import datetime

class RedisUtils(object):
    
    

    def __init__(self, host="localhost", port=6379, db=0):
        self.host = host
        self.port = port 
        self.db = db
        self.model_name = None
        self.redis = redis.Redis(host, port, db)
        self.values = {}
        self.info = {}
        
    def initialize(self, model_name):
        self.model_name = model_name
        self.update_info()
        self.update_values()


    def update_info(self):
        self.info = {}
        self.info["model_info"] = json.loads( self.redis.get( self.model_name + '|model_info'))
        self.info["certificate_info"] = json.loads( self.redis.get( self.model_name + '|certificate_info'))
        self.info["metric_info"] = json.loads( self.redis.get( self.model_name + '|metric_info'))
        

    def update_values(self):
        self.values = {}
        self.values["metric_values"]=[]
        for data in  self.redis.lrange( self.model_name + '|metric_values', 0, -1):
            self.values["metric_values"].append( json.loads(data))
        
        self.values["certificate_values"] = []
        for data in self.redis.lrange( self.model_name + '|certificate_values', 0, -1):
            self.values["certificate_values"].append( json.loads(data))

 