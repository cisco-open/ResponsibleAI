import pandas as pd
import scipy.stats
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


def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (xTest == ai.dataset.data_dict["test"].X).all()
    assert (yTest == ai.dataset.data_dict["test"].y).all()
    assert (xTrain == ai.dataset.data_dict["train"].X).all()
    assert (yTrain == ai.dataset.data_dict["train"].y).all()


def test_num_nan_rows():
    """Tests that the RAI num nan rows calculation is correct."""
    xTestDf = pd.DataFrame(xTest, columns=features)
    assert metrics['summary_stats']['num_nan_rows'] == xTestDf.shape[0] - xTestDf.dropna().shape[0]


def test_percent_Nan_rows():
    """Tests that the RAI percent nan rows calculation is correct."""
    xTestDf = pd.DataFrame(xTest, columns=features)
    assert metrics['summary_stats']['percent_nan_rows'] == (xTestDf.shape[0] - xTestDf.copy().dropna().shape[0])/len(xTestDf)


def test_kstat_1():
    """Tests that the RAI kstat_1 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert metrics['summary_stats']['kstat_1'][i] == scipy.stats.kstat(xTest[:, i], 1)
        else:
            assert metrics['summary_stats']['kstat_1'][i] is None


def test_kstat_2():
    """Tests that the RAI kstat_2 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert metrics['summary_stats']['kstat_2'][i] == scipy.stats.kstat(xTest[:, i], 2)
        else:
            assert metrics['summary_stats']['kstat_2'][i] is None


def test_kstat_3():
    """Tests that the RAI kstat_3 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert metrics['summary_stats']['kstat_3'][i] == scipy.stats.kstat(xTest[:, i], 3)
        else:
            assert metrics['summary_stats']['kstat_3'][i] is None


def test_kstat_4():
    """Tests that the RAI kstat_1 calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert metrics['summary_stats']['kstat_4'][i] == scipy.stats.kstat(xTest[:, i], 4)
        else:
            assert metrics['summary_stats']['kstat_4'][i] is None


def test_kstatvar():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert metrics['summary_stats']['kstatvar'][i] == scipy.stats.kstatvar(xTest[:, i])
        else:
            assert metrics['summary_stats']['kstatvar'][i] is None


def test_iqr():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            assert metrics['summary_stats']['iqr'][i] == scipy.stats.iqr(xTest[:, i])
        else:
            assert metrics['summary_stats']['iqr'][i] is None


def test_bayes_mvs():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            mean, var, std = scipy.stats.bayes_mvs(xTest[:, i])
            assert metrics['summary_stats']['bayes_mean_avg'][i] == mean[0]
            assert metrics['summary_stats']['bayes_variance_avg'][i] == var[0]
            assert metrics['summary_stats']['bayes_std_avg'][i] == std[0]
        else:
            assert metrics['summary_stats']['bayes_mean_avg'][i] is None
            assert metrics['summary_stats']['bayes_variance_avg'][i] is None
            assert metrics['summary_stats']['bayes_std_avg'][i] is None


def test_frozen_mvs():
    """Tests that the RAI kstatvar calculation is correct."""
    for i, feature in enumerate(features):
        if not feature.categorical:
            mean, var, std = scipy.stats.mvsdist(xTest[:, i])
            assert metrics['summary_stats']['frozen_mean_mean'][i] == mean.mean()
            assert metrics['summary_stats']['frozen_mean_variance'][i] == mean.var()
            assert metrics['summary_stats']['frozen_mean_std'][i] == mean.std()

            assert metrics['summary_stats']['frozen_variance_mean'][i] == var.mean()
            assert metrics['summary_stats']['frozen_variance_variance'][i] == var.var()
            assert metrics['summary_stats']['frozen_variance_std'][i] == var.std()

            assert metrics['summary_stats']['frozen_std_mean'][i] == std.mean()
            assert metrics['summary_stats']['frozen_std_variance'][i] == std.var()
            assert metrics['summary_stats']['frozen_std_std'][i] == std.std()
        else:
            assert metrics['summary_stats']['frozen_mean_mean'][i] is None
            assert metrics['summary_stats']['frozen_mean_variance'][i] is None
            assert metrics['summary_stats']['frozen_mean_std'][i] is None

            assert metrics['summary_stats']['frozen_variance_mean'][i] is None
            assert metrics['summary_stats']['frozen_variance_variance'][i] is None
            assert metrics['summary_stats']['frozen_variance_std'][i] is None

            assert metrics['summary_stats']['frozen_std_mean'][i] is None
            assert metrics['summary_stats']['frozen_std_variance'][i] is None
            assert metrics['summary_stats']['frozen_std_std'][i] is None
