from sklearn.utils import check_X_y
from sklearn.neighbors import NearestNeighbors
from RAI.metrics.metric_group import MetricGroup
from RAI.metrics.ai360_helper.AI360_helper import *
import numpy as np
import pandas as pd

__all__ = ['compatibility']

 
# Log loss, roc and brier score have been removed. s

_config = {
    "name" : "individual_fairness",
    "compatibility" : {"type_restriction": "classification", "output_restriction": "choice"},
    "src": "equal_treatment",
    "dependency_list": [],
    "tags": ["fairness","Equal Treatment"],
    "complexity_class": "linear",
    "metrics": {
        "generalized_entropy_error": {
            "display_name": "Generalized Entropy Error",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "between_group_generalized_entropy_error": {
            "display_name": "Between Group Generalized Entropy Error",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "theil_index": {
            "display_name": "Theil Index",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "coefficient_of_variation": {
            "display_name": "Coefficient of Variance",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        },
        "consistency_score": {
            "display_name": "Consistency Score",
            "type": "numeric",
            "tags": [],
            "has_range": True,
            "range": [0, 2],
            "explanation": ""
        }
    }
}


class IndividualFairnessMetricGroup(MetricGroup, config=_config):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        if "data" and "predictions" in data_dict:
            data = data_dict["data"]
            preds = data_dict["predictions"]
            priv_group = None
            if self.ai_system.user_config is not None and "equal_treatment" in self.ai_system.user_config and "priv_group" in self.ai_system.user_config["equal_treatment"]:
                priv_group = self.ai_system.user_config["equal_treatment"]["priv_group"]

            y = _convert_to_ai360(self, data)
            # MAY REQUIRE ADJUSTMENT DEPENDING ON AI360'S USE.
            self.metrics['generalized_entropy_error'].value = _generalized_entropy_error(y, preds, pos_label=priv_group[1])
            self.metrics['between_group_generalized_entropy_error'].value = _between_group_generalized_entropy_error(y, preds, prot_attr=priv_group[0], pos_label= priv_group[1])
            self.metrics['theil_index'].value = _theil_index(_get_b(y, preds, 1))
            self.metrics['coefficient_of_variation'].value = _coefficient_of_variation(_get_b(y, preds, 1))
            self.metrics['consistency_score'].value = _consistency_score(data.X, data.y)


def _convert_to_ai360(metric_group, data):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features]
    df = pd.DataFrame(data.X, columns=names)
    df['y'] = data.y
    X, y = standardize_dataset(df, prot_attr=['race'], target='y')
    return y


# CUSTOM FUNCTION TO WORK WITH AI360. REVIEW THE MATH.
def _get_b(y_true, y_pred, pos_label):
    return 1 + (y_pred == pos_label) - (y_true == pos_label)


def _generalized_entropy_error(y_true, y_pred, alpha=2, pos_label=1):
    b = 1 + (y_pred == pos_label) - (y_true == pos_label)
    return _generalized_entropy_index(b, alpha=alpha)


def _between_group_generalized_entropy_error(y_true, y_pred, prot_attr=None, priv_group=None, alpha=2, pos_label=1):
    groups, _ = check_groups(y_true, prot_attr)
    b = np.empty_like(y_true, dtype='float')
    if priv_group is not None:
        groups = [1 if g == priv_group else 0 for g in groups]
    for g in np.unique(groups):
        b[groups == g] = (1 + (y_pred[groups == g] == pos_label)
                            - (y_true[groups == g] == pos_label)).mean()
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