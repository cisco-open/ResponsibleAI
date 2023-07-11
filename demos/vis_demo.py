# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# importing modules
import os
import sys
import inspect

from dotenv import load_dotenv

from tensorflow.keras.datasets import cifar10

# importing RAI modules
from RAI.AISystem import AISystem
from RAI.dataset import NumpyData, Dataset
from RAI.db.service import RaiDB

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

load_dotenv(f'{currentdir}/../.env')

# Configuration
use_dashboard = True 

# Get Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset = Dataset({"train": NumpyData(x_train, y_train), "test": NumpyData(x_test, y_test)})

# set configuration
configuration = {}

ai = AISystem(name="cifar-10", dataset=dataset)
ai.initialize(user_config=configuration)

if use_dashboard:
    r = RaiDB(ai)
    r.reset_data()
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")
