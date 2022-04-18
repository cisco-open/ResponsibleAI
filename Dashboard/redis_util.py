import os
import json
import redis
import sys
import datetime
import functools

class RedisUtils(object):
    
    

    def __init__(self, host="localhost", port=6379, db=0):
        self.host = host
        self.port = port 
        self.db = db
        self.model_name = None
        self.redis = redis.Redis(host, port, db)
        self.redis_pub = self.redis.pubsub()
        

        self.values = {}
        self.info = {}
        self.initialized = False
        self.subscribers={}


    def initialize(self, model_name, subscribers = None):
        if subscribers is None:
            subscribers = set()
        self.subscribers = {}
        for item in subscribers:
            self.subscribers[item] = False
        
        if self.initialized:
            return
         
        self.model_name = model_name
        
        def sub_handler( msg):
            print("a new message received: ", msg)
            self.update_info()
            self.update_values()
            for item in self.subscribers:
                self.subscribers[item] = True
        
        try:
            print("channel subsribed", model_name+"|update")
            self.redis_pub.subscribe( **{ model_name+"|update": sub_handler} )
            self.redis_pub.run_in_thread(sleep_time=.1)
        except:
            print("unable to subscribe to redis pub/sub")
        
        
        self.update_info()
        self.update_values()
        self.initialized = True

    def close(self):
        if self.redis_pub is not None:
            try:
                self.redis_pub.unsubscribe(self.model_name+"|update")
                self.redis_pub.close()
            except:
                print("Problem in closing pub/sub")
        self.redis.close()
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

 