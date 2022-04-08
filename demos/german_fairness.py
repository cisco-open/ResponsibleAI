from demo_helper_code.demo_helper_functions import *
# TODO:
# DONE.     Change show date show it defaults to showing index not date.
# add more certificates, use accuracy for performance.
# Move to jupyter.


# add margin to graph, choose more instructive names for the models. - Main page.
# Change name to Area under curve to AUC-ROC reciever.
# Drop down with dates to select when you want to look at certificates.
# replace performance on main page with balanced accuracy.
# Area under precision recall. - Add that.
# Check if CLEVER needs to be 1hot.

# 3-4 minutes. 2 mis tops for the other two.

# 4.5 MINUTES AVERAGE.


# First round, trained model performance good fairness bad. Step by step.
# Load dataset, using sklearn to train a model, typical datascience task.
# Time to evaluate model to get insight.
# Show certificates/badges in flow. Show relevant information to show something failed.


#
#
# Demo structure:
# Explain task and dataset.
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
dataset_metadata, X, y, xTrain, xTest, yTrain, yTest = get_german_dataset()


# Represent the database in RAIs format.
rai_MetaDatabase, rai_fairness_config = get_rai_metadatabase(dataset_metadata)
rai_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)


# Create a classifier and train it on the dataset.
reg, train_preds, test_preds = get_classifier_and_preds(xTrain, xTest, yTrain)


# Represent the AI Model and the Dataset as a RAI Ai System
credit_ai = get_german_rai_ai_system(reg, rai_fairness_config, rai_MetaDatabase, rai_dataset)


# Compute Metrics
credit_ai.reset_redis()
credit_ai.compute_metrics(test_preds, data_type="test", export_title="Original")

# View metrics of the predictions and dataset.
credit_ai.viewGUI()


# Reweigh data to accommodate for the bias against age
xTrain, xTest, yTrain, yTest = reweigh_dataset_for_age(X, y)


# Create a RAI dataset from the new reweighed data and replace the existing dataset with the new one.
new_reweigh_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)
credit_ai.dataset = new_reweigh_dataset


# Retrain classifier on the new reweighed data
reg, train_preds, test_preds = get_classifier_and_preds(xTrain, xTest, yTrain, reg=reg)


# Recompute Metrics
credit_ai.compute_metrics(test_preds, data_type="test", export_title="Reweigh")





# View GUI
credit_ai.viewGUI()


