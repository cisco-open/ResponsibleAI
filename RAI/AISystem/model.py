__all__ = ['Model', 'model_caps']

model_caps = ['predict', 'prob']

# this is an abstract interface for models
class Model:
    def __init__(self) -> None:
        pass
    def predict(self, X):
        pass
    def prob(self, X):
        pass
