import os
import sys
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import Data, Dataset
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI
from sklearn.ensemble import RandomForestClassifier
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
use_dashboard = True


# Get Dataset
data_path = "../data/adult/"
train_data = pd.read_csv(data_path + "train.csv", header=0, skipinitialspace=True, na_values="?")
test_data = pd.read_csv(data_path + "test.csv", header=0, skipinitialspace=True, na_values="?")
all_data = pd.concat([train_data, test_data], ignore_index=True)


# Get X and y data, as well as RAI Meta information from the Dataframe
rai_meta_information, X, y, rai_output_feature = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar")


# Create Data Splits and pass them to RAI
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})


# Create Model and RAIs representation of it
clf = RandomForestClassifier(n_estimators=4, max_depth=6)
model = Model(agent=clf, output_features=rai_output_feature, name="cisco_income_ai", predict_fun=clf.predict,
              predict_prob_fun=clf.predict_proba, description="Income Prediction AI", model_class="RFC")


# Create RAI AISystem to pass all relevant data to RAI
ai = AISystem(name="income_classification_reg",  task='binary_classification', meta_database=rai_meta_information,
              dataset=dataset, model=model)
configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                              "protected_attributes": ["race"], "positive_label": 1},
                 "time_complexity": "polynomial"}
ai.initialize(user_config=configuration)


# Train the model, generate predictions
clf.fit(xTrain, yTrain)
test_predictions = clf.predict(xTest)


# Pass predictions to RAI
ai.compute({"test": {"predict": test_predictions}}, tag='model')


# Connect to the Dashboard
r = RaiRedis(ai)
r.connect()
r.reset_redis()
r.add_measurement()
r.export_metadata()
