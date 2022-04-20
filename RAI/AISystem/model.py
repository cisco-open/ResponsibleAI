__all__ = ['Model', 'model_caps']
model_caps = ['predict', 'prob']


# this is an abstract interface for models
class Model:
    def __init__(self, agent=None,  run_func=None, loss_function=None, optimizer=None, name=None, display_name=None, model_class=None, adaptive=None) -> None:
        
        self._name = name
        self._display_name = display_name
        if display_name is None:
            self._display_name = name
        self.model_class = model_class
        self.adaptive = adaptive
        self.agent = agent
        self.run_func = run_func
        self.loss_function = loss_function
        self.optimizer=optimizer

    def predict(self, x):
        pass
        # return agent.run_func(x)

    def prob(self, x):
        pass
