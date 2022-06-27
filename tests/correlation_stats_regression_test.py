import scipy.stats
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

np.random.seed(21)
x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})

features = [
    Feature("MedInc", 'float', "Median Income"),
    Feature("HouseAge", 'float', "Median House age in Block Group"),
    Feature("AveRooms", 'float', "Average number of rooms per household"),
    Feature("AveBedrms", 'float', "Average number of bedrooms per household"),
    Feature("Population", 'float', "Block group population"),
    Feature("AveOccup", 'float', "Average Number of Household members"),
    Feature("Latitude", 'float', "Block group Latitude"),
    Feature("Longitude", 'float', "Block group Longitude")
]
meta = MetaDatabase(features)

reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", task='regression', meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

ai.compute({"test": {"predict": predictions}}, tag="regression")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_pearson_correlation():
    """Tests that the RAI pearson correlation calculation is correct."""
    for i in range(len(features)):
        assert metrics['correlation_stats_regression']['pearson_correlation'][i] == \
               scipy.stats.pearsonr(xTest[:, i], yTest)


def test_spearman_correlation():
    """Tests that the RAI spearman correlation calculation is correct."""
    for i in range(len(features)):
        assert metrics['correlation_stats_regression']['spearman_correlation'][i] == \
               scipy.stats.spearmanr(xTest[:, i], yTest)
