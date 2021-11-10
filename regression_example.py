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
# Function to compare our result to sklearn's result.

def test_metric(res, actual, preds):
    if 'class_bias' in res:
        print("Testing class_bias metrics")
        assert res['class_bias']['accuracy'] == sklearn.metrics.accuracy_score(actual, preds)
        assert res['class_bias']['balanced_accuracy'] == sklearn.metrics.balanced_accuracy_score(actual, preds)
        assert res['class_bias']['f1-single'] == sklearn.metrics.f1_score(actual, preds, average="macro")
        assert res['class_bias']['jaccard_score-single'] == sklearn.metrics.jaccard_score(actual, preds, average="macro")
        assert np.array_equal(res['class_bias']['confusion_matrix'], sklearn.metrics.confusion_matrix(actual, preds))
    if 'reg_bias' in res:
        print("Testing class_reg metrics")
        assert res['reg_bias']['explained_variance'] == sklearn.metrics.explained_variance_score(actual, preds)
        assert res['reg_bias']['mean_absolute_error'] == sklearn.metrics.mean_absolute_error(actual, preds)
        assert res['reg_bias']['mean_absolute_percentage_error'] == sklearn.metrics.mean_absolute_percentage_error(actual, preds)
        assert res['reg_bias']['mean_gamma_deviance'] == sklearn.metrics.mean_gamma_deviance(actual, preds)
        assert res['reg_bias']['mean_poisson_deviance'] == sklearn.metrics.mean_poisson_deviance(actual, preds)
        assert res['reg_bias']['mean_squared_error'] == sklearn.metrics.mean_squared_error(actual, preds)
        assert res['reg_bias']['mean_squared_log_error'] == sklearn.metrics.mean_squared_log_error(actual, preds)
        assert res['reg_bias']['median_absolute_error'] == sklearn.metrics.median_absolute_error(actual, preds)
        assert res['reg_bias']['r2'] == sklearn.metrics.r2_score(actual, preds)


# Compute Metrics Using our Engine
res = ai.get_metric_values()

# Test Metric Values
print("\nTESTING Metrics:")
test_metric(res, yTest, model_preds)


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

