__all__ = ['Model', 'task_types']
task_types = ["binary_classification", "multiclass_classification", "regression"]


class Model:
    def __init__(self, task=None, name=None, display_name=None, agent=None, loss_function=None, optimizer=None, model_class=None, description=None, predict_fun=None, predict_prob_fun=None, generate_text_fun=None) -> None:
        self.task = task
        self.name = name
        self.display_name = name
        if display_name is not None:
            self.display_name = display_name
        self.agent = agent
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_class = model_class
        self.description = description
        self.predict_fun = predict_fun
        self.predict_prob_fun = predict_prob_fun
        self.generate_text_fun = generate_text_fun
