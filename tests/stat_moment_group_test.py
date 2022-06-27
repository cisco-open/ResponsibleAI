import scipy.stats
from numpy.testing import assert_almost_equal
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
model = Model(agent=reg, predict_fun=reg.predict, name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", task='regression', meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

ai.compute({"test": predictions}, tag="regression")

metrics = ai.get_metric_values()
metrics = metrics["test"]
info = ai.get_metric_info()


def test_moment_1():
    """Tests that the RAI moment 1 calculation is correct."""
    assert_almost_equal(scipy.stats.moment(xTest, 1), metrics['stat_moment_group']['moment_1'], 6)


def test_moment_2():
    """Tests that the RAI moment 2 calculation is correct."""
    assert_almost_equal(scipy.stats.moment(xTest, 2), metrics['stat_moment_group']['moment_2'], 6)


def test_moment_3():
    """Tests that the RAI moment 3 calculation is correct."""
    assert_almost_equal(scipy.stats.moment(xTest, 3), metrics['stat_moment_group']['moment_3'], 6)
