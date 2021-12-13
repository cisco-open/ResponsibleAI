from demo_helper_code.demo_helper_functions import *

# Get Dataset
xTrain, xTest, yTrain, yTest = load_breast_cancer_dataset()


# Put Data into RAI's format
rai_MetaDatabase = get_breast_cancer_metadatabase()
rai_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)


# Create Various Tree models trained on the data
forest_model, decision_tree, gradient_boost = get_ai_trees(xTrain, yTrain)


# Place model in RAI format.
ai_tree_comparison = get_breast_cancer_rai_ai_system(forest_model, None, None, rai_MetaDatabase, rai_dataset, "cert_list_ad_demo.json")


# Compute Metrics for each model.
ai_tree_comparison.compute_metrics(forest_model.predict(xTest), data_type="test", export_title="Random Forest")


# Exchange model for Decision Tree and Recompute Metrics
ai_tree_comparison.task.model.agent = decision_tree
ai_tree_comparison.compute_metrics(decision_tree.predict(xTest), data_type="test", export_title="Decision Tree")


# Exchange model for Extra Trees and Recompute Metrics
ai_tree_comparison.task.model.agent = gradient_boost
ai_tree_comparison.compute_metrics(gradient_boost.predict(xTest), data_type="test", export_title="Grad Boost")


ai_tree_comparison.viewGUI()
