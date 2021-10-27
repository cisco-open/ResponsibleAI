import RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task


# defining metadata, this should also be possible by loading a dictionary or reading from a json file later
# Some do this through pandas DF.
features = [
    Feature("x1", 'float32', "first column"),
    Feature("x2", 'float32', "second column"),
    Feature("x3", 'float32', "third column") 
]
meta = MetaDatabase(features)
configuration = {"bias": {"sensitive_features": ["second column"]}}

# creating data record for training

dataset_x = [[1, 3, 5], [1, 2, 3], [2, 4, 6], [5, 13, 11], [2, 2, 2]]  # Get dataset
dataset_y = [0, 2, 1, 0, 1]
training_data = Data(dataset_x, dataset_y)

# creating a dataset object, currently no validation or testing data
dataset = Dataset(training_data)


# creating a model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(dataset_x, dataset_y)
# model = Model(clf)


# creating a task
task = Task(model=clf, type='regression')

print("Creating AI System")
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
ai.initialize()


ai.compute_metrics()
print(ai.export_metrics_values())


