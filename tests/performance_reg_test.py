from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sklearn
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


def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (xTest == ai.dataset.data_dict["test"].X).all()
    assert (yTest == ai.dataset.data_dict["test"].y).all()
    assert (xTrain == ai.dataset.data_dict["train"].X).all()
    assert (yTrain == ai.dataset.data_dict["train"].y).all()


def test_explained_variance():
    """Tests that the RAI explained variance calculation is correct."""
    assert metrics['performance_reg']['explained_variance'] == sklearn.metrics.explained_variance_score(yTest, predictions)


def test_mean_absolute_error():
    """Tests that the RAI mean absolute error calculation is correct."""
    assert metrics['performance_reg']['mean_absolute_error'] == sklearn.metrics.mean_absolute_error(yTest, predictions)


def test_mean_absolute_percentage_error():
    """Tests that the RAI mean absolute error calculation is correct."""
    assert metrics['performance_reg']['mean_absolute_percentage_error'] == sklearn.metrics.mean_absolute_percentage_error(yTest, predictions)


def test_mean_gamma_deviance():
    """Tests that the RAI mean gamma deviance calculation is correct."""
    assert metrics['performance_reg']['mean_gamma_deviance'] == sklearn.metrics.mean_gamma_deviance(yTest, predictions)


def test_mean_poisson_deviance():
    """Tests that the RAI mean poisson deviance calculation is correct."""
    assert metrics['performance_reg']['mean_poisson_deviance'] == sklearn.metrics.mean_poisson_deviance(yTest, predictions)


def test_mean_squared_error():
    """Tests that the RAI mean squared error calculation is correct."""
    assert metrics['performance_reg']['mean_squared_error'] == sklearn.metrics.mean_squared_error(yTest, predictions)


def test_mean_squared_log_error():
    """Tests that the RAI mean squared log error calculation is correct."""
    assert metrics['performance_reg']['mean_squared_log_error'] == sklearn.metrics.mean_squared_log_error(yTest, predictions)


def test_median_absolute_error():
    """Tests that the RAI mean squared log error calculation is correct."""
    assert metrics['performance_reg']['median_absolute_error'] == sklearn.metrics.median_absolute_error(yTest, predictions)


def test_r2():
    """Tests that the RAI r2 calculation is correct."""
    assert metrics['performance_reg']['r2'] == sklearn.metrics.r2_score(yTest, predictions)
