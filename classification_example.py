import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
import sklearn.metrics
import numpy as np


# Get Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
x, y = load_breast_cancer(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

nums = np.ones((xTrain.shape[0], 1))
nums[:int(xTrain.shape[0]/2)] = 0
xTrain = np.hstack((xTrain, nums))
nums = np.ones((xTest.shape[0], 1))
nums[:int(xTest.shape[0]/2)] = 0
xTest = np.hstack((xTest, nums))

# Set up features
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis",
                "race"]
features = []
for feature in features_raw:
    features.append(Feature(feature, "float32", feature))

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
configuration = {"equal_treatment": {"priv_group": ("race", 1)}}
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
ai.initialize()

# Train model
reg.fit(xTrain, yTrain)
model_preds = reg.predict(xTest)

# Make Predictions
ai.compute_metrics(model_preds)
# Function to compare our result to sklearn's result.


def test_metric(res, actual, preds):
    # res is the result from ai_system.get_metric_values()
    if 'class_bias' in res: # Check the metrics in the class_bias metric group
        print("Testing class_bias metrics")
        assert res['class_bias']['accuracy'] == sklearn.metrics.accuracy_score(actual, preds)
        assert res['class_bias']['balanced_accuracy'] == sklearn.metrics.balanced_accuracy_score(actual, preds)
        assert res['class_bias']['f1-single'] == sklearn.metrics.f1_score(actual, preds, average="macro")
        assert res['class_bias']['jaccard_score-single'] == sklearn.metrics.jaccard_score(actual, preds, average="macro")
        assert np.array_equal(res['class_bias']['confusion_matrix'], sklearn.metrics.confusion_matrix(actual, preds))
    if 'reg_bias' in res:  # Check th emetrics in the reg_bias metric groups
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

for group in res:
    for metric in res[group]:
        print(metric, " = ", res[group][metric])


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

# Demonstrating Searching
query = "Bias"
print("\nSearching Metrics for ", query)
result = ai.search(query)
print(result)

print("\nSummarizing Results")
ai.summarize()

print("\nViewing GUI")
ai.viewGUI()
