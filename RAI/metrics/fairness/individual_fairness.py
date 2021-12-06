from sklearn.utils import check_X_y
from sklearn.neighbors import NearestNeighbors
from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import numpy as np
import pandas as pd

__all__ = ['compatibility']

 
# Log loss, roc and brier score have been removed. s

_config = {
    "name": "individual_fairness",
    "compatibility": {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness", "Individual Fairness"],
    "complexity_class": "linear",
    "metrics": {
        "generalized_entropy_error": {
            "display_name": "Generalized Entropy Error",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Measures inequality over a population. Lower values are better."
        },
        "theil_index": {
            "display_name": "Theil Index",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Measures the entropic distances between real data and an ideal and equal population. Lower values are better."
        },
        "coefficient_of_variation": {
            "display_name": "Coefficient of Variance",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, None],
            "explanation": "Two times the square root of the generalzed entropy index with a=2. "
        },
        "consistency_score": {
            "display_name": "Consistency Score",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 1],
            "explanation": "Measures how similiar labels are for similiar instances. R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, “Learning Fair Representations,” International Conference on Machine Learning, 2013."
        }
    }
}


class IndividualFairnessMetricGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def is_compatible(ai_system):
        compatible = _config["compatibility"]["type_restriction"] is None \
                    or ai_system.task.type == _config["compatibility"]["type_restriction"] \
                    or ai_system.task.type == "binary_classification" and _config["compatibility"]["type_restriction"] == "classification"
        compatible = compatible \
                     and "fairness" in ai_system.user_config \
                     and "protected_attributes" in ai_system.user_config["fairness"] \
                     and "positive_label" in ai_system.user_config["fairness"]
        return compatible

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            prot_attr = []
            pos_label = 1
            if self.ai_system.user_config is not None and "fairness" in self.ai_system.user_config and "priv_group" in \
                    self.ai_system.user_config["fairness"]:
                prot_attr = self.ai_system.user_config["fairness"]["protected_attributes"]
                pos_label = self.ai_system.user_config["fairness"]["positive_label"]

            y = _convert_to_ai360(self, data, prot_attr)
            # MAY REQUIRE ADJUSTMENT DEPENDING ON AI360'S USE.
            self.metrics['generalized_entropy_error'].value = _generalized_entropy_error(y, preds, pos_label=pos_label)
            self.metrics['theil_index'].value = _theil_index(_get_b(y, preds, 1))
            self.metrics['coefficient_of_variation'].value = _coefficient_of_variation(_get_b(y, preds, 1))
            self.metrics['consistency_score'].value = _consistency_score(data.X, data.y)


def _convert_to_ai360(metric_group, data, prot_attr):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df = pd.DataFrame(data.X, columns=names)
    df['y'] = data.y
    X, y = standardize_dataset(df, prot_attr=prot_attr, target='y')
    return y


# CUSTOM FUNCTION TO WORK WITH AI360. REVIEW THE MATH.
def _get_b(y_true, y_pred, pos_label):
    return 1 + (y_pred == pos_label) - (y_true == pos_label)


def _generalized_entropy_error(y_true, y_pred, alpha=2, pos_label=1):
    b = 1 + (y_pred == pos_label) - (y_true == pos_label)
    return _generalized_entropy_index(b, alpha=alpha)


def _theil_index(b):
    return _generalized_entropy_index(b, alpha=1)


def _coefficient_of_variation(b):
    return 2 * np.sqrt(_generalized_entropy_index(b, alpha=2))


def _consistency_score(X, y, n_neighbors=5):
    # cast as ndarrays
    X, y = check_X_y(X, y)
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nbrs.fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)

    # compute consistency score
    return 1 - abs(y - y[indices].mean(axis=1)).mean()


def _generalized_entropy_index(b, alpha=2):
    if alpha == 0:
        return -(np.log(b / b.mean()) / b.mean()).mean()
    elif alpha == 1:
        # moving the b inside the log allows for 0 values
        return (np.log((b / b.mean())**b) / b.mean()).mean()
    else:
        return ((b / b.mean())**alpha - 1).mean() / (alpha * (alpha - 1))