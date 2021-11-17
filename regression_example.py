import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
import sklearn.metrics
import numpy as np


# Get Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

# Hook data in with our Representation
training_data = Data(xTest, yTest)  # Accepts Data and GT
dataset = Dataset(training_data)  # Accepts Training, Test and Validation Set

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
model = Model(agent=reg, name="Cisco RealEstate AI", model_class="Random Forest Regressor", adaptive=False)
# Indicate the task of the model
task = Task(model=model, type='regression')

# Create AISystem from previous objects. AISystems are what users will primarily interact with.

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=None)
ai.initialize()

# Train model
reg.fit(xTrain, yTrain)
model_preds = reg.predict(xTest)

# Make Predictions
ai.compute_metrics(model_preds)

# Dictionary used for testing metrics
# Function to compare our result to sklearn's result.

# Compute Metrics Using our Engine


# Compute Metrics Using our Engine
resv_f = ai.get_metric_values_flat()
resv_d = ai.get_metric_values_dict()
resi_f = ai.get_metric_info_flat()
resi_d = ai.get_metric_info_dict()

for key in resv_f:
    if hasattr(resv_f[key], "__len__"):
        # print(resi_f[key]['display_name'], " = ", 'list ...')
        print(resi_f[key]['display_name'], " = ", resv_f[key])
    else:
        print(resi_f[key]['display_name'], " = ", resv_f[key])

# Getting Metric Information
print("\nGetting Metric Information")
metric_info = ai.get_metric_info_flat()
for metric in metric_info:
    print(metric_info[metric])

# Get Model Information
print("\nGetting Model Info:")
res = ai.get_model_info()
print(res)

# Demonstrating Searching
query = "Bias"
print("\nSearching Metrics for ", query)
result = ai.search(query)
print(result)

# print("\nSummarizing Results")
# ai.summarize()

# reset all previous keys
# ai.reset_redis()

# export to redis
# ai.export_data_flat()


print("\nViewing GUI")
# ai.viewGUI()
