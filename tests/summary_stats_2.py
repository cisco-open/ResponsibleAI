from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

use_dashboard = False
np.random.seed(21)

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})

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

reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, predict_fun=reg.predict, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", task='regression', meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

ai.compute({"test": {"predict": predictions}}, tag="regression")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()

for g in metrics:
    for m in metrics[g]:
        if g!="tree_model_metadata":
            print(g, " ", m, " ", metrics[g][m])