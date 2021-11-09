import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
import sklearn.metrics
import numpy as np


# Get Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
x, y = load_breast_cancer(return_X_y=True)
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
            "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
features = []
for feature in features_raw:
    features.append(Feature(feature, "float32", feature))

xTrain, xTest, yTrain, yTest = train_test_split(x, y)

# Hook data in with our Representation
training_data = Data(xTest, yTest)  # Accepts Data and GT
dataset = Dataset(training_data)  # Accepts Training, Test and Validation Set
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
model = Model(agent=reg, name="Cisco Breast Cancer AI", model_class="Random Forest Classifier", adaptive=False)
# Indicate the task of the model
task = Task(model=model, type='classification')

# Create AISystem from previous objects. AISystems are what users will primarily interact with.
configuration = {"bias": {"args": {"accuracy": {"normalize": True}}, "sensitive_features": ["second column"]}}
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
ai.initialize()

# Train model
reg.fit(xTrain, yTrain)
model_preds = reg.predict(xTest)

# Make Predictions
ai.compute_metrics(model_preds)

# Dictionary used for testing metrics
test_group = {"accuracy": [sklearn.metrics.accuracy_score, {}], "balanced_accuracy": [sklearn.metrics.balanced_accuracy_score, {}],
             "f1-single": [sklearn.metrics.f1_score, {"average": "macro", "labels": [0, 1]}],
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

# Demonstrating Searching
query = "Bias"
print("\nSearching Metrics for ", query)
result = ai.search(query)
print(result)

print("\nSummarizing Results")
ai.summarize()
