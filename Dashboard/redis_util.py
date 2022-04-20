import os
import json
import redis
import sys
import datetime
import functools
from collections import defaultdict
class RedisUtils(object):
    
    

    def __init__(self, host="localhost", port=6379, db=0):
        self._host = host
        self._port = port 
        self._db = db
        
        
        

        self.values = {}
        self.info = {}
        self._initialized = False
        self._subscribers = defaultdict(bool)

        
        self._current_project = None
        self._current_project_name = None

    def has_update(self, channel, reset = True):
        
        val =  self._subscribers[channel]
        if reset:
            self.reset_channel(channel)
        return val


    def reset_channel(self, channel):
        self._subscribers[channel] = False

    def _init_pubsub(self):
        
        def sub_handler( msg):
            print("a new message received: ", msg)
            if self._current_project_name:
                self._update_info()
                self._update_values()
            for item in self._subscribers:
                self._subscribers[item] = True
        
        self._redis_pub = self._redis.pubsub()
        
        try:
            print("channel subsribed", self._model_name+"|update")
            self._redis_pub.subscribe( **{ self._model_name+"|update": sub_handler} )
            # self._redis_pub.run_in_thread(sleep_time=.1)
        except:
            print("unable to subscribe to redis pub/sub")



    def initialize(self, subscribers = None):
        
        if self._initialized:
            return

        self._redis = redis.Redis(self._host, self._port, self._db)

        self._subscribers = {}
        if subscribers is not None:
            for item in subscribers:
                self._subscribers[item] = False
        
        

        self._init_pubsub()
        
        
        
        self._update_projects()
        self._initialized = True

    def _reload(self):
        
        self._update_projects()
        self._update_info()
        self._update_values()


    def close(self):
        if self._redis_pub is not None:
            try:
                self._redis_pub.unsubscribe(self._model_name+"|update")
                self._redis_pub.close()
            except:
                print("Problem in closing pub/sub")
        self._redis.close()
    

    def get_project_info(self):
        return self._current_project["project_info"]
    def get_metric_info(self):
            return self._current_project["metric_info"]
    def get_certificate_info(self):
            return self._current_project["certificate_info"]
    def get_certificate_values(self):
            return self._current_project["certificate_values"]
    def get_metric_values(self):
            return self._current_project["metric_values"]

    def set_current_project(self, project_name):
        print("changing from", self._current_project_name, "to", project_name)
        if self._current_project_name==project_name:
            return
        self._current_project_name = project_name
        self._current_project = {}
        self._update_info()
        self._update_values()

    def _update_projects(self):
        self._projects = self._redis.smembers("projects")
        self._projects = [ s.decode('utf-8') for s in self._projects]   
    def get_projects_list(self):
        return self._projects

    def _update_info(self):
        self.info = {}
        print(self._current_project_name)
        self._current_project["project_info"] = \
            json.loads( self._redis.get( self._current_project_name + '|project_info'))
        self._current_project["certificate_info"] = \
            json.loads( self._redis.get( self._current_project_name + '|certificate_info'))
        self._current_project["metric_info"] = \
            json.loads( self._redis.get( self._current_project_name + '|metric_info'))
        

    def _update_values(self):
        self.values = {}
        self._current_project["metric_values"]=[]
        for data in  self._redis.lrange( self._current_project_name + '|metric_values', 0, -1):
            self._current_project["metric_values"].append( json.loads(data))
        
        self._current_project["certificate_values"] = []
        for data in self._redis.lrange( self._current_project_name + '|certificate_values', 0, -1):
            self._current_project["certificate_values"].append( json.loads(data))

    def __del__(self):
        pass
    # self.close()