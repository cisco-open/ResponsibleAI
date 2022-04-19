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
        self._model_name = None
        
        

        self.values = {}
        self.info = {}
        self._initialized = False
        self._subscribers = defaultdict(bool)


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



    def initialize(self, model_name, subscribers = None):
        
        if self._initialized:
            return

        self._model_name = model_name
        self._redis = redis.Redis(self._host, self._port, self._db)

        self._subscribers = {}
        if subscribers is not None:
            for item in subscribers:
                self._subscribers[item] = False
        
        

        self._init_pubsub()
        
        
        
        
        self._update_info()
        self._update_values()
        self._initialized = True

    def close(self):
        if self._redis_pub is not None:
            try:
                self._redis_pub.unsubscribe(self._model_name+"|update")
                self._redis_pub.close()
            except:
                print("Problem in closing pub/sub")
        self._redis.close()
    
    def _update_info(self):
        self.info = {}
        self.info["model_info"] = json.loads( self._redis.get( self._model_name + '|model_info'))
        self.info["certificate_info"] = json.loads( self._redis.get( self._model_name + '|certificate_info'))
        self.info["metric_info"] = json.loads( self._redis.get( self._model_name + '|metric_info'))
        

    def _update_values(self):
        self.values = {}
        self.values["metric_values"]=[]
        for data in  self._redis.lrange( self._model_name + '|metric_values', 0, -1):
            self.values["metric_values"].append( json.loads(data))
        
        self.values["certificate_values"] = []
        for data in self._redis.lrange( self._model_name + '|certificate_values', 0, -1):
            self.values["certificate_values"].append( json.loads(data))

    def __del__(self):
        pass
    # self.close()