from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
import numpy as np


# Get Pandas Dataframe dataset.
#
# To test RAI, we will use the German Credit Dataset.
#
# German Credit Dataset:
# Dataset for a bank to predict whether or not someone is a good or bad credit risk for a loan.
#
# Input Features:
# Bank Account status, Duration of bank account, total credit history, loan purpose, total credit amount,
# savings account size, Age, Installment plants, Housing, Current Bank credits, Job skill level type,
# Number of liable people, has telephone, is foreign worker
#
# Output:
# 0 (Bad), 1 (Good)



# Collect data stored in pandas dataframe
from demo_helper_code.demo_helper_functions import get_german_dataset
df_info = get_german_dataset() # df_info is a dictionary containing the dataframe and information about its features.
pandas_df = df_info["df"]
print("German Credit Data: \n", str(pandas_df.head()))


# Put data in format where predictions can be made.
from sklearn.model_selection import train_test_split
y = pandas_df.pop("credit")
X = pandas_df
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=2, stratify=y)
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()
yTrain = yTrain.to_numpy()
yTest = yTest.to_numpy()


# Get Feature Metadata used in RAI from Pandas Dataframe.
from RAI.utils import df_to_meta_database
# Helper function which collects all relevant meta information from a dataframe.
meta, fairness_config = df_to_meta_database(X, categorical_values=df_info["categorical_meanings"],
                                            protected_attribute_names=df_info["protected_attribute_names"],
                                            privileged_info=df_info["privileged_info"],
                                            positive_label=df_info["positive_label"])


# Put the train and test data into RAI's representation.
from RAI.dataset import Data, Dataset
from RAI.AISystem import AISystem, Model, Task
training_data = Data(xTrain, yTrain)  # Accepts Data and GT
test_data = Data(xTest, yTest)
dataset = Dataset(training_data, test_data=test_data)  # Accepts Training, Test and Validation Set


# Create an AI model and train it to make predictions.
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)
test_preds = reg.predict(xTest)


# Create an AI System using all collected information to allow for metric computation.
model = Model(agent=reg, name="cisco_german_fairness", display_name="Cisco German Fairness", model_class="Random Forest Classifier", adaptive=False)
task = Task(model=model, type='binary_classification', description="Predict the credit score of various Germans.")
configuration = {"fairness": fairness_config, "time_complexity": "linear"}
credit_ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration, custom_certificate_location="RAI\\certificates\\standard\\cert_list_credit.json")
credit_ai.initialize()


# Compute Metrics
credit_ai.reset_redis()
credit_ai.compute_metrics(test_preds, data_type="test")
credit_ai.export_data_flat("Original")


# Compute Certificates
credit_ai.compute_certificates()
credit_ai.export_certificates("Original")


# Viewing results for these, we can clearly see that nearly all fairness certificates fail for age.
# To accomodate for this we can reweigh the dataset using tools from:
# F. Kamiran and T. Calders,  "Data Preprocessing Techniques for Classification without Discrimination," Knowledge and Information Systems, 2012
# And measure the disparate impact of the retrained model.


# Reweigh data to accomodate for bias
from demo_helper_code.demo_helper_functions import reweigh_dataset_for_age
xTrain, xTest, yTrain, yTest = reweigh_dataset_for_age(X, y)


# Replace the current dataset with the reweighed dataset.
training_data = Data(xTrain, yTrain)
test_data = Data(xTest, yTest)
dataset = Dataset(training_data, test_data=test_data)
credit_ai.dataset = dataset


# Retrain classifier
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

