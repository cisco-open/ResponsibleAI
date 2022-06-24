from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

use_dashboard = False
np.random.seed(21)

# Hook data in with our Representation
dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})

# Indicate the features of the dataset (Columns)
features = [
    Feature("MedInc", 'float', "Median Income", categorical=False),
    Feature("HouseAge", 'float', "Median House age in Block Group", categorical=False),
    Feature("AveRooms", 'float', "Average number of rooms per household", categorical=False),
    Feature("AveBedrms", 'float', "Average number of bedrooms per household", categorical=False),
    Feature("Population", 'float', "Block group population", categorical=False),
    Feature("AveOccup", 'float', "Average Number of Household members", categorical=False),
    Feature("Latitude", 'float', "Block group Latitude", categorical=False),
    Feature("Longitude", 'float', "Block group Longitude", categorical=False)
]
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, task='regression', predict_fun=reg.predict, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

# Create AISystem from previous objects. AISystems are what users will primarily interact with.

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

# Train model
reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

# Make Predictions
ai.compute({"test": predictions}, tag="regression")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()

for g in metrics:
    for m in metrics[g]:
        if g!="tree_model_metadata":
            print(g, " ", m, " ", metrics[g][m])
        '''
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])
        '''