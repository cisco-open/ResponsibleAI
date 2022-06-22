import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
import scipy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
x, y = load_breast_cancer(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

val_0_count = 20

nums = np.ones((xTrain.shape[0], 1))
nums[:val_0_count] = 0
xTrain = np.hstack((xTrain, nums))
xTrain = np.hstack((xTrain, nums))

nums = np.ones((xTest.shape[0], 1))
nums[:val_0_count] = 0
xTest = np.hstack((xTest, nums))
xTest = np.hstack((xTest, nums))

# Set up features
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
features = []

for feature in features_raw:
    features.append(Feature(feature, "float32", feature))
features.append(Feature("race", "integer", "race value", categorical=True, values={0:"black", 1:"white"}))
features.append(Feature("gender", "integer", "race value", categorical=True, values={1:"male", 0:"female"}))

# Hook data in with our Representation
training_data = Data(xTrain, yTrain)  # Accepts Data and GT
test_data = Data(xTest, yTest)
dataset = Dataset({"train": training_data, "test": test_data})  # Accepts Training, Test and Validation Set
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
model = Model(agent=rfc, task='binary_classification', name="cisco_cancer_ai", display_name="Cisco Health AI",
              model_class="Random Forest Classifier", description="Detect Cancer in patients using skin measurements")

# Create AISystem from previous objects. AISystems are what users will primarily interact with.
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1}}
ai = AISystem("cancer_detection", meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

# Train model
rfc.fit(xTrain, yTrain)
predictions = rfc.predict(xTest)

# Make Predictions
ai.compute({"test": predictions}, tag="binary classification")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


for g in metrics:
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])


def test_point_biserial_r():
    """Tests that the RAI relfreq calculation is correct."""
    for i in range(len(features_raw)):
        assert metrics['correlation_stats_binary']['point-biserial-r'][i] == \
               scipy.stats.pointbiserialr(xTest[:, i], yTest)

# TODO: Should be from binary X values to continuous y values
