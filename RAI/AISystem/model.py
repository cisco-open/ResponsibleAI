__all__ = ['Model', 'model_caps']
model_caps = ['predict', 'prob']


# this is an abstract interface for models
class Model:
    def __init__(self, agent=None, run_func=None, name=None, display_name=None, model_class=None, adaptive=None) -> None:
        if ' ' in name:
            raise Exception("Please enter a name with no spaces")
        if name is None:
            raise Exception("Please enter a valid name")
        self.name = name
        self.display_name = display_name
        if display_name is None:
            self.display_name = name
        self.model_class = model_class
        self.adaptive = adaptive
        self.agent = agent
        self.run_func = run_func

    def predict(self, x):
        pass
        # return agent.run_func(x)

    def prob(self, x):
        pass
