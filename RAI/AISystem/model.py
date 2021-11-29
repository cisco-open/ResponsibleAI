__all__ = ['Model', 'model_caps']
model_caps = ['predict', 'prob']


# this is an abstract interface for models
class Model:
    def __init__(self, agent=None, run_func=None, name=None, model_class=None, adaptive=None) -> None:
        if ' ' in name:
            raise Exception("Please enter a name with no spaces")
        self.name = name
        self.model_class = model_class
        self.adaptive = adaptive
        self.agent = agent
        self.run_func = run_func

    def predict(self, x):
        pass
        # return agent.run_func(x)

    def prob(self, x):
        pass
