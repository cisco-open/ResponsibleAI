import pandas as pd
import scipy.stats

import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
import numpy as np

# Get Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sklearn
x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

use_dashboard = False
np.random.seed(21)

# Hook data in with our Representation
dataset = Dataset(train_data=Data(xTrain, yTrain), test_data=Data(xTest, yTest))

# Indicate the features of the dataset (Columns)
features = [
    Feature("MedInc", 'float32', "Median Income"),
    Feature("HouseAge", 'float32', "Median House age in Block Group"),
    Feature("AveRooms", 'float32', "Average number of rooms per household"),
    Feature("AveBedrms", 'float32', "Average number of bedrooms per household"),
    Feature("Population", 'float32', "Block group population"),
    Feature("AveOccup", 'float32', "Average Number of Household members"),
    Feature("Latitude", 'float32', "Block group Latitude"),
    Feature("Longitude", 'float32', "Block group Longitude")
]
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor", adaptive=False)
# Indicate the task of the model
task = Task(model=model, type='regression')

# Create AISystem from previous objects. AISystems are what users will primarily interact with.

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", meta_database=meta, dataset=dataset, task=task, enable_certificates=False)
ai.initialize(user_config=configuration)

# Train model
reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

# Make Predictions
ai.compute(predictions, data_type='test', tag="regression")

metrics = ai.get_metric_values()
info = ai.get_metric_info()

for g in metrics:
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])
