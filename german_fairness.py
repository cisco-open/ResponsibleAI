from demo_helper_code.demo_helper_functions import *
# TODO:
# Change show date show it defaults to showing index not date.
# add margin to graph, choose more instructive names for the models.
# Drop down with dates to select when you want to look at certificates.
# Move to jupyter.
# introduce data and task -> high level fairness indicator,
#
# Demo structure:
# Explain task an dataset.
# Show naive approach and evaluation technique
# Show RAI captures more complex issues with data, and what causes it.
# Fix data because we know now of the bias. Recompute metrics.
# Improved fairness
# RAI allows us to detect these issues and give us tools to better evaluate models.

'''
We will use RAI to create and evaluate a model on the German Credit Dataset.

German Credit Dataset:
    Dataset for a bank to predict whether or not someone is a good or bad credit risk for a loan.

    Input Features:
        Bank Account status, Duration of bank account, total credit history, loan purpose, total credit amount,
        savings account size, Age, Installment plants, Housing, Current Bank credits, Job skill level type,
        Number of liable people, has telephone, is foreign worker

    Output:
        0 (Bad), 1 (Good)
'''


# Get German Credit Database
meta_info, X, y, xTrain, xTest, yTrain, yTest = get_german_dataset()


# Put the data in a RAI Dataset Object so that RAI can compute metrics
rai_MetaDatabase, rai_fairness_config = get_rai_metadatabase(meta_info)
rai_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)


# Create an AI model and train it to make predictions.
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)
test_preds = reg.predict(xTest)


# Create an AI System using all collected information to allow for metric computation.
from RAI.AISystem import AISystem, Model, Task
model = Model(agent=reg, name="cisco_german_fairness", display_name="Cisco German Fairness", model_class="Random Forest Classifier", adaptive=False)
task = Task(model=model, type='binary_classification', description="Predict the credit score of various Germans.")
configuration = {"fairness": rai_fairness_config, "time_complexity": "linear"}
credit_ai = AISystem(meta_database=rai_MetaDatabase, dataset=rai_dataset, task=task, user_config=configuration, custom_certificate_location="RAI\\certificates\\standard\\cert_list_credit.json")
credit_ai.initialize()


# Compute Metrics
credit_ai.reset_redis()
credit_ai.compute_metrics(test_preds, data_type="test")
credit_ai.export_data_flat("Original")


# Compute Certificates
credit_ai.compute_certificates()
credit_ai.export_certificates("Original")


# View performance on dataset, biased against age.
# credit_ai.viewGUI()


# Reweigh data to accommodate for the bias against age
from demo_helper_code.demo_helper_functions import reweigh_dataset_for_age
xTrain, xTest, yTrain, yTest = reweigh_dataset_for_age(X, y)


# Replace the current dataset with the reweighed dataset.
reweigh_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)
credit_ai.dataset = reweigh_dataset


# Retrain classifier on reweighed data
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)
test_preds = reg.predict(xTest)


# Recompute Metrics
credit_ai.compute_metrics(test_preds, data_type="test")
credit_ai.export_data_flat("Weight")


# Compute Certificates
credit_ai.compute_certificates()
credit_ai.export_certificates("Weight")


# View GUI
credit_ai.viewGUI()


