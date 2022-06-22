from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sklearn


x, y = fetch_california_housing(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)
use_dashboard = False
np.random.seed(21)

# Hook data in with our Representation
dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})

# Indicate the features of the dataset (Columns)
features = [
    Feature("MedInc", 'float32', "Median Income"),
    Feature("HouseAge", 'float32', "Median House age in Block Group"),
    Feature("AveRooms", 'float32', "Average number of rooms per household"),
    Feature("AveBedrms", 'float32', "Average number of bedrooms per household"),
    Feature("Population", 'float32', "Block group population"),
    Feature("AveOccup", 'float32', "Average Number of Household members"),
    Feature("Latitude", 'float32', "Block group Latitude"),
    Feature("Longitude", 'float32', "Block group Longitude")
]
meta = MetaDatabase(features)

# Create a model to make predictions
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=15, max_depth=20)
model = Model(agent=reg, task='regression', name="Cisco_RealEstate_AI", model_class="Random Forest Regressor")

# Create AISystem from previous objects. AISystems are what users will primarily interact with.

configuration = {"equal_treatment": {"priv_groups": [("Gender", 1)]}}
ai = AISystem("Regression example", meta_database=meta, dataset=dataset, model=model, enable_certificates=False)
ai.initialize(user_config=configuration)

# Train model
reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

# Make Predictions
ai.compute(predictions, data_type='test', tag="regression")

metrics = ai.get_metric_values()
info = ai.get_metric_info()

for g in metrics:
    for m in metrics[g]:
        if "type" in info[g][m]:
            if info[g][m]["type"] in ("numeric", "vector-dict", "text"):
                print(g, m, metrics[g][m])


def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (xTest == ai.dataset.test_data.X).all()
    assert (yTest == ai.dataset.test_data.y).all()
    assert (xTrain == ai.dataset.train_data.X).all()
    assert (yTrain == ai.dataset.train_data.y).all()


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
