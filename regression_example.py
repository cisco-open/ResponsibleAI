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
    Feature("Longitude", 'float32', "BLock group Longitude")
]
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, name="Cisco RealEstate AI", model_class="Random Forest Regressor", adaptive=False)
# Indicate the task of the model
task = Task(model=model, type='regression')

# Create AISystem from previous objects. AISystems are what users will primarily interact with.
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=None)
ai.initialize()

# Train model
reg.fit(xTrain, yTrain)
model_preds = reg.predict(xTest)

# Make Predictions
ai.compute_metrics(model_preds)

# Dictionary used for testing metrics
test_group = {"accuracy": [sklearn.metrics.accuracy_score, {}], "balanced_accuracy": [sklearn.metrics.balanced_accuracy_score, {}],
             "f1-single": [sklearn.metrics.f1_score, {"average": "macro", "labels": [0, 1, 2]}],
             "jaccard_score-single": [sklearn.metrics.jaccard_score, {"average": "macro"}], "confusion_matrix": [sklearn.metrics.confusion_matrix, {}],
             "explained_variance": [sklearn.metrics.explained_variance_score, {}], "mean_absolute_error": [sklearn.metrics.mean_absolute_error, {}],
             "mean_absolute_percentage_error": [sklearn.metrics.mean_absolute_percentage_error, {}], "mean_gamma_deviance": [sklearn.metrics.mean_gamma_deviance, {}],
             "mean_poisson_deviance": [sklearn.metrics.mean_poisson_deviance, {}], "mean_squared_error": [sklearn.metrics.mean_squared_error, {}],
             "mean_squared_log_error": [sklearn.metrics.mean_squared_log_error, {}], "median_absolute_error": [sklearn.metrics.median_absolute_error, {}],
             "r2": [sklearn.metrics.r2_score, {}]}


# Function to compare our result to sklearn's result.
def test_metric(metric_name, function, preds, actual, expected):
    print("\tTesting ", metric_name)
    if isinstance(expected, np.ndarray):
        if not np.array_equal(function[0](actual, preds, **function[1]), expected):
            print("Test Failed")
    else:
        # print("\t", expected, " = ", function[0](actual, preds, **function[1]))
        if function[0](actual, preds, **function[1]) != expected:
            print("Test Failed")


# Compute Metrics Using our Engine
ai.compute_metrics(preds=model_preds)
res = ai.get_metric_values()

# Test Metric Values
print("\nTESTING Metrics:")
for group in res:
    for metric in res[group]:
        if metric in test_group:
            ans = res[group][metric]  # Get the output of the metric value.
            if isinstance(ans, list):
                ans = ans[1]
            test_metric(metric, test_group[metric], model_preds, yTest, res[group][metric]) # Compare our score to sklearns.


# Getting Metric Information
print("\nGetting Metric Information")
res = ai.get_metric_info()
metrics = res["metrics"]
for metric in metrics:
    print(metrics[metric])

# Get Metric Categories
print("\nGetting Metric Categories")
categories = res["categories"]
for category in categories:
    print(category, " ", categories[category])

# Get Model Information
print("\nGetting Model Info:")
res = ai.get_model_info()
print(res)

