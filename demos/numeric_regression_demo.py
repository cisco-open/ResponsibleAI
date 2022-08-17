import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from RAI.dataset import NumpyData, Dataset
from RAI.AISystem import AISystem, Model
import numpy as np
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

np.random.seed(21)
result = fetch_california_housing(as_frame=True)
target = result.target
df = result.data
df[target.name] = target

meta, X, y, output = df_to_RAI(df, target_column="MedHouseVal")
xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1)


use_dashboard = True


reg = RandomForestRegressor(n_estimators=15, max_depth=20)

model = Model(agent=reg, output_features=output, name="cisco_income_ai", predict_fun=reg.predict,
              description="Income Prediction AI", model_class="Random Forest Regressor", )
configuration = {"time_complexity": "polynomial"}

dataset = Dataset({"train": NumpyData(xTrain, yTrain), "test": NumpyData(xTest, yTest)})
ai = AISystem(name="AdultDB_regression",  task='regression', meta_database=meta, dataset=dataset, model=model)
ai.initialize(user_config=configuration)

reg.fit(xTrain, yTrain)
predictions = reg.predict(xTest)

ai.compute({"test": {"predict": predictions}}, tag="regression")

if use_dashboard:
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
    r.add_measurement()

ai.display_metric_values("test")

from RAI.Analysis import AnalysisManager

analysis = AnalysisManager()
print("available analysis: ", analysis.get_available_analysis(ai, "test"))
# result = analysis.run_analysis(ai, ["test"], ["FairnessAnalysis"])
result = analysis.run_all(ai, "test", "Test run!")
for analysis in result:
    print("Analysis: " + analysis)
    print(result[analysis].to_string())

