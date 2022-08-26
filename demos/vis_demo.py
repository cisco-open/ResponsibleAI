import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from RAI.AISystem import AISystem
from RAI.dataset import NumpyData, Dataset
from RAI.redis import RaiRedis

from tensorflow.keras.datasets import cifar10

use_dashboard = True 

# Get Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset = Dataset({"train": NumpyData(x_train, y_train), "test": NumpyData(x_test, y_test)})

# set configuration
configuration = {}

ai = AISystem(name="cifar-10", dataset=dataset)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
