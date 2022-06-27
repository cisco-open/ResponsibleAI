__all__ = ['Model']
from RAI.metrics.metric_group import all_task_types


class Model:
    def __init__(self, predict_fun=None, predict_prob_fun=None, generate_text_fun=None, name=None, display_name=None, agent=None, loss_function=None, optimizer=None, model_class=None, description=None) -> None:
        assert name is not None, "Please provide a model name"
        self.output_types = {}
        self.predict_fun = predict_fun
        if predict_fun is not None:
            self.output_types["predict"] = predict_fun
        self.predict_prob_fun = predict_prob_fun
        if predict_prob_fun is not None:
            self.output_types["predict_proba"] = predict_prob_fun
        self.generate_text_fun = generate_text_fun
        if generate_text_fun is not None:
            self.output_types["generate_text"] = generate_text_fun
        self.name = name
        self.display_name = name
        if display_name is not None:
            self.display_name = display_name
        self.agent = agent
        self.input_type = None
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_class = model_class
        self.description = description
