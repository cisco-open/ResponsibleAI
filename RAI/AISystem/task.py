__all__ = ['Task', 'task_types' ]


from .model import *
task_types = ["binary_classification", "multiclass_classification", "regression" ] 


# task is an abstraction for general machine learning tasks, 
class Task:
    def __init__(self, model, type, description="") -> None:
        self.model = model
        self.type = type
        self.description = description

    def predict(self, X):
        return self.model.predict(X)

    def prob(self, X):
        return self.model.prob(X)