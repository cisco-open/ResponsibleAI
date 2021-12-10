# add more certificates, use accuracy for performance.



# Get Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
x, y = load_breast_cancer(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=2)


# Put Data into RAI's format
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
training_data = Data(xTrain, yTrain)  # Accepts Data and GT
test_data = Data(xTest, yTest)
dataset = Dataset(training_data, test_data=test_data)  # Accepts Training, Test and Validation Set


# Create Meta database to store feature values
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
features = []
for feature in features_raw:
    features.append(Feature(feature, "float32", feature))
meta = MetaDatabase(features)


# Create AI Model, make predictions
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)
test_preds = reg.predict(xTest)


# Place model in RAI format.
model = Model(agent=reg, name="cisco_tree_ai", display_name="Cisco Tree AIs", model_class="Random Forest Classifier", adaptive=False)
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {}
ai_tree_comparison = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration, custom_certificate_location="RAI\\certificates\\standard\\cert_list_ad_demo.json")
ai_tree_comparison.initialize()


# Compute Metrics
ai_tree_comparison.reset_redis()
ai_tree_comparison.compute_metrics(test_preds, data_type="test")
ai_tree_comparison.export_data_flat("Random Forest")


# Compute Certificates
ai_tree_comparison.compute_certificates()
ai_tree_comparison.export_certificates("Random Forest")


# We have now computed metrics for one model type, but want to compare it to others.
# We will now create new models and compute metrics for them to compare to.


# Create ExtraTreesClassifier model to compare to:
from sklearn.ensemble import ExtraTreesClassifier
reg = ExtraTreesClassifier(n_estimators=4, max_depth=6, random_state=0)
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)
test_preds = reg.predict(xTest)


# Exchange the AI System model with an ExtraTreesClassifier
ai_tree_comparison.task.model.agent = reg


# Recompute Metrics
ai_tree_comparison.compute_metrics(test_preds, data_type="test")
ai_tree_comparison.export_data_flat("Extra Trees")


# Compute Certificates
ai_tree_comparison.compute_certificates()
ai_tree_comparison.export_certificates("Extra Trees")


# We will now finally compare these two models to a GradientBoost model.
# To compare, we will once again create an instance of the model and compute metrics for it.


# Create GradientBoost model to compare to:
from sklearn.ensemble import GradientBoostingClassifier
reg = GradientBoostingClassifier(n_estimators=3, max_depth=6, random_state=0)
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)
test_preds = reg.predict(xTest)


# Exchange AI System model
ai_tree_comparison.task.model.agent = reg


# Recompute Metrics
ai_tree_comparison.compute_metrics(test_preds, data_type="test")
ai_tree_comparison.export_data_flat("Boosting")


# Compute Certificates
ai_tree_comparison.compute_certificates()
ai_tree_comparison.export_certificates("Boosting")


# Visually compare results.
ai_tree_comparison.viewGUI()

