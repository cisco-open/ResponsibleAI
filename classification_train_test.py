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

nums = np.ones((xTrain.shape[0], 1))
nums[:int(xTrain.shape[0]/2)] = 0
xTrain = np.hstack((xTrain, nums))



nums = np.ones((xTest.shape[0], 1))
nums[:int(xTest.shape[0]/2)] = 0
xTest = np.hstack((xTest, nums))

nums = np.ones((xTest.shape[0], 1))
nums[:int(xTest.shape[0]/2)] = 0
xTest = np.hstack((xTest, nums))

# Set up features
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
features = []

for feature in features_raw:
    features.append(Feature(feature, "float32", feature))
features.append(Feature("race", "integer", "race value", categorical=True, values=[{0:"black"}, {1:"white"}]))
features.append(Feature("gender", "integer", "race value", categorical=True, values=[{1:"male"}, {0:"female"}]))

# Hook data in with our Representation
training_data = Data(xTrain, yTrain)  # Accepts Data and GT
test_data = Data(xTest, yTest)
dataset = Dataset(training_data, test_data=test_data)  # Accepts Training, Test and Validation Set
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
model = Model(agent=reg, name="cisco_cancer_ai", display_name="Cisco Health AI", model_class="Random Forest Classifier", adaptive=False)
# Indicate the task of the model
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")

# Create AISystem from previous objects. AISystems are what users will primarily interact with.
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1}}
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
ai.initialize()

# Train model
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)

# Make Predictions
ai.reset_redis()
ai.compute_metrics(train_preds, data_type="train")
ai.export_data_flat("Train set")
ai.export_certificates()


test_preds = reg.predict(xTest)
# Make Predictions




ai.compute_metrics(test_preds, data_type="test")
ai.export_data_flat("Test set")
ai.export_certificates()



print("\nViewing GUI")
ai.viewGUI()
print("DONE")



