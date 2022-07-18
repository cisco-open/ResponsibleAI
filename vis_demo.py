import pandas as pd
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import Data, Dataset
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI

from tensorflow.keras.datasets import cifar10
import numpy as np

use_dashboard = True 

# Get Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset = Dataset({"train": Data(x_train, y_train), "test": Data(x_test, y_test)})

# set configuration
configuration = {}

ai = AISystem(name="cifar-10", dataset=dataset)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()