import warnings

from collections import namedtuple
from pandas.core.dtypes.common import is_list_like

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer as _make_scorer, recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y
from sklearn.exceptions import UndefinedMetricWarning



__all__ = [
    'standardize_dataset', 'check_groups','check_already_dropped',
    'difference', 'ratio',
    'make_scorer',
    # helpers
    'specificity_score', 'base_rate', 'selection_rate', 'generalized_fpr',
    'generalized_fnr',
    'recall_score'
]


def specificity_score(y_true, y_pred, pos_label=1, sample_weight=None):
    """Compute the specificity or true negative rate.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    """
    MCM = multilabel_confusion_matrix(y_true, y_pred, labels=[pos_label],
                                      sample_weight=sample_weight)
    tn, fp, fn, tp = MCM.ravel()
    negs = tn + fp
    if negs == 0:
        warnings.warn('specificity_score is ill-defined and being set to 0.0 '
                      'due to no negative samples.', UndefinedMetricWarning)
        return 0.
    return tn / negs


def generalized_fnr(y_true, probas_pred, pos_label=1, sample_weight=None):
    r"""Return the ratio of generalized false negatives to positive examples in
    the dataset, :math:`GFNR = \tfrac{GFN}{P}`.
    Generalized confusion matrix measures such as this are calculated by summing
    the probabilities of the positive class instead of the hard predictions.
    Args:
        y_true (array-like): Ground-truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Generalized false negative rate. If there are no positive samples
        in y_true, this will raise an
        :class:`~sklearn.exceptions.UndefinedMetricWarning` and return 0.
    """
    idx = (y_true == pos_label)
    if not np.any(idx):
        warnings.warn("generalized_fnr is ill-defined because there are no "
                      "positive samples in y_true.", UndefinedMetricWarning)
        return 0.
    if sample_weight is None:
        return 1 - probas_pred[idx].mean()
    return 1 - np.average(probas_pred[idx], weights=sample_weight[idx])


def generalized_fpr(y_true, probas_pred, pos_label=1, sample_weight=None):
    r"""Return the ratio of generalized false positives to negative examples in
    the dataset, :math:`GFPR = \tfrac{GFP}{N}`.
    Generalized confusion matrix measures such as this are calculated by summing
    the probabilities of the positive class instead of the hard predictions.
    Args:
        y_true (array-like): Ground-truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Generalized false positive rate. If there are no negative samples
        in y_true, this will raise an
        :class:`~sklearn.exceptions.UndefinedMetricWarning` and return 0.
    """
    idx = (y_true != pos_label)
    if not np.any(idx):
        warnings.warn("generalized_fpr is ill-defined because there are no "
                      "negative samples in y_true.", UndefinedMetricWarning)
        return 0.
    if sample_weight is None:
        return probas_pred[idx].mean()
    return np.average(probas_pred[idx], weights=sample_weight[idx])






def make_scorer(score_func, is_ratio=False, **kwargs):
    """Make a scorer from a 'difference' or 'ratio' metric (e.g.
    :func:`statistical_parity_difference`).
    Args:
        score_func (callable): A ratio/difference metric with signature
            ``score_func(y, y_pred, **kwargs)``.
        is_ratio (boolean, optional): Indicates if the metric is ratio or
        difference based.
    """
    if is_ratio:

        def score(y, y_pred, **kwargs):
            ratio = score_func(y, y_pred, **kwargs)
            eps = np.finfo(float).eps
            ratio_inverse = 1 / ratio if ratio > eps else eps
            return min(ratio, ratio_inverse)

        scorer = _make_scorer(score, **kwargs)
    else:

        def score(y, y_pred, **kwargs):
            diff = score_func(y, y_pred, **kwargs)
            return abs(diff)

        scorer = _make_scorer(score, greater_is_better=False, **kwargs)
    return scorer

def base_rate(y_true, y_pred=None, pos_label=1, sample_weight=None):
    r"""Compute the base rate, :math:`Pr(Y = \text{pos_label}) = \frac{P}{P+N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Base rate.
    """
    idx = (y_true == pos_label)
    return np.average(idx, weights=sample_weight)


def ratio(func, y, *args, prot_attr=None, priv_group=1, sample_weight=None,
          **kwargs):
    """Compute the ratio between unprivileged and privileged subsets for an
    arbitrary metric.
    Note: The optimal value of a ratio is 1. To make it a scorer, one must
    take the minimum of the ratio and its inverse.
    Unprivileged group is taken to be the inverse of the privileged group.
    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics.metrics`.
        y (pandas.Series): Outcome vector with protected attributes as index.
        *args: Additional positional args to be passed through to func.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        **kwargs: Additional keyword args to be passed through to func.
    Returns:
        scalar: Ratio of metric values for unprivileged and privileged groups.
    """
    groups, _ = check_groups(y, prot_attr)
    idx = (groups == priv_group)
    unpriv = map(lambda a: a[~idx], (y,) + args)
    priv = map(lambda a: a[idx], (y,) + args)
    if sample_weight is not None:
        numerator = func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
        denominator = func(*priv, sample_weight=sample_weight[idx], **kwargs)
    else:
        numerator = func(*unpriv, **kwargs)
        denominator = func(*priv, **kwargs)

    if denominator == 0:
        warnings.warn("The ratio is ill-defined and being set to 0.0 because "
                      "'{}' for privileged samples is 0.".format(func.__name__),
                      UndefinedMetricWarning)
        return 0.

    return numerator / denominator


def selection_rate(y_true, y_pred, pos_label=1, sample_weight=None):
    r"""Compute the selection rate, :math:`Pr(\hat{Y} = \text{pos_label}) =
    \frac{TP + FP}{P + N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values. Ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Selection rate.
    """
    return base_rate(y_pred, pos_label=pos_label, sample_weight=sample_weight)


def check_groups(arr, prot_attr, ensure_binary=False):
    """Get groups from the index of arr.
    If there are multiple protected attributes provided, the index is flattened
    to be a 1-D Index of tuples. If ensure_binary is ``True``, raises a
    ValueError if there are not exactly two unique groups. Also checks that all
    provided protected attributes are in the index.
    Args:
        arr (:class:`pandas.Series` or :class:`pandas.DataFrame`): A Pandas
            object containing protected attribute information in the index.
        prot_attr (single label or list-like): Protected attribute(s). If
            ``None``, all protected attributes in arr are used.
        ensure_binary (bool): Raise an error if the resultant groups are not
            binary.
    Returns:
        tuple:
            * **groups** (:class:`pandas.Index`) -- Label (or tuple of labels)
              of protected attribute for each sample in arr.
            * **prot_attr** (`list-like`) -- Modified input. If input is a
              single label, returns single-item list. If input is ``None``
              returns list of all protected attributes.
    """
    if not hasattr(arr, 'index'):
        raise TypeError(
                "Expected `Series` or `DataFrame`, got {} instead.".format(
                        type(arr).__name__))

    all_prot_attrs = [name for name in arr.index.names if name]  # not None or ''
    if prot_attr is None:
        prot_attr = all_prot_attrs
    elif not is_list_like(prot_attr):
        prot_attr = [prot_attr]

    if any(p not in arr.index.names for p in prot_attr):
        raise ValueError("Some of the attributes provided are not present "
                         "in the dataset. Expected a subset of:\n{}\nGot:\n"
                         "{}".format(all_prot_attrs, prot_attr))

    groups = arr.index.droplevel(list(set(arr.index.names) - set(prot_attr)))
    groups = groups.to_flat_index()

    n_unique = groups.nunique()
    if ensure_binary and n_unique != 2:
        raise ValueError("Expected 2 protected attribute groups, got {}".format(
                groups.unique() if n_unique > 5 else n_unique))

    return groups, prot_attr


def standardize_dataset(df, prot_attr, target, sample_weight=None, usecols=[],
                       dropcols=[], numeric_only=False, dropna=True):
    """Separate data, targets, and possibly sample weights and populate
    protected attributes as sample properties.
    Args:
        df (pandas.DataFrame): DataFrame with features and target together.
        prot_attr (single label or list-like): Label or list of labels
            corresponding to protected attribute columns. Even if these are
            dropped from the features, they remain in the index.
        target (single label or list-like): Column label of the target (outcome)
            variable.
        sample_weight (single label, optional): Name of the column containing
            sample weights.
        usecols (single label or list-like, optional): Column(s) to keep. All
            others are dropped.
        dropcols (single label or list-like, optional): Column(s) to drop.
        numeric_only (bool): Drop all non-numeric, non-binary feature columns.
        dropna (bool): Drop rows with NAs.
    Returns:
        collections.namedtuple:
            A tuple-like object where items can be accessed by index or name.
            Contains the following attributes:
            * **X** (`pandas.DataFrame`) -- Feature array.
            * **y** (`pandas.DataFrame` or `pandas.Series`) -- Target array.
            * **sample_weight** (`pandas.Series`, optional) -- Sample weights.
    Note:
        The order of execution for the dropping parameters is: numeric_only ->
        usecols -> dropcols -> dropna.
    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['X', 'y', 'Z'])
        >>> train = standardize_dataset(df, prot_attr='Z', target='y')
        >>> reg = LinearRegression().fit(*train)
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> df = pd.DataFrame(np.hstack(make_classification(n_features=5)))
        >>> X, y = standardize_dataset(df, prot_attr=0, target=5)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    """
    orig_cols = df.columns
    if numeric_only:
        for col in df.select_dtypes('category'):
            if df[col].cat.ordered:
                df[col] = df[col].factorize(sort=True)[0]
                df[col] = df[col].replace(-1, np.nan)
        df = df.select_dtypes(['number', 'bool'])
    nonnumeric = orig_cols.difference(df.columns)

    prot_attr = check_already_dropped(prot_attr, nonnumeric, 'prot_attr')
    if len(prot_attr) == 0:
        raise ValueError("At least one protected attribute must be present.")
    df = df.set_index(prot_attr, drop=False, append=True)

    target = check_already_dropped(target, nonnumeric, 'target')
    if len(target) == 0:
        raise ValueError("At least one target must be present.")
    y = pd.concat([df.pop(t) for t in target], axis=1).squeeze()  # maybe Series

    # Column-wise drops
    orig_cols = df.columns
    if usecols:
        usecols = check_already_dropped(usecols, nonnumeric, 'usecols')
        df = df[usecols]
    unused = orig_cols.difference(df.columns)

    dropcols = check_already_dropped(dropcols, nonnumeric, 'dropcols', warn=False)
    dropcols = check_already_dropped(dropcols, unused, 'dropcols', 'usecols', False)
    df = df.drop(columns=dropcols)

    # Index-wise drops
    if dropna:
        notna = df.notna().all(axis=1) & y.notna()
        df = df.loc[notna]
        y = y.loc[notna]

    if sample_weight is not None:
        return namedtuple('WeightedDataset', ['X', 'y', 'sample_weight'])(
                          df, y, df.pop(sample_weight).rename('sample_weight'))
    return namedtuple('Dataset', ['X', 'y'])(df, y)


class ColumnAlreadyDroppedWarning(UserWarning):
    """Warning used if a column is attempted to be dropped twice."""

def check_already_dropped(labels, dropped_cols, name, dropped_by='numeric_only',
                          warn=True):
    """Check if columns have already been dropped and return only those that
    haven't.
    Args:
        labels (single label or list-like): Column labels to check.
        dropped_cols (set or pandas.Index): Columns that were already dropped.
        name (str): Original arg that triggered the check (e.g. dropcols).
        dropped_by (str, optional): Original arg that caused dropped_cols``
            (e.g. numeric_only).
        warn (bool, optional): If ``True``, produces a
            :class:`ColumnAlreadyDroppedWarning` if there are columns in the
            intersection of dropped_cols and labels.
    Returns:
        list: Columns in labels which are not in dropped_cols.
    """
    if not is_list_like(labels):
        labels = [labels]
    str_labels = [c for c in labels if isinstance(c, str)]
    already_dropped = dropped_cols.intersection(str_labels)
    if warn and any(already_dropped):
        warnings.warn("Some column labels from `{}` were already dropped by "
                "`{}`:\n{}".format(name, dropped_by, already_dropped.tolist()),
                ColumnAlreadyDroppedWarning, stacklevel=2)
    return [c for c in labels if not isinstance(c, str) or c not in already_dropped]


def difference(func, y, *args, prot_attr=None, priv_group=1, sample_weight=None,
               **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.
    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set greater_is_better to False.
    Unprivileged group is taken to be the inverse of the privileged group.
    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics.metrics`.
        y (pandas.Series): Outcome vector with protected attributes as index.
        *args: Additional positional args to be passed through to func.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        **kwargs: Additional keyword args to be passed through to func.
    Returns:
        scalar: Difference in metric value for unprivileged and privileged
        groups.
    Examples:
        >>> X, y = fetch_german(numeric_only=True)
        >>> y_pred = LogisticRegression().fit(X, y).predict(X)
        >>> difference(precision_score, y, y_pred, prot_attr='sex',
        ... priv_group='male')
        -0.06955430006277463
    """
    groups, _ = check_groups(y, prot_attr)
    idx = (groups == priv_group)
    unpriv = map(lambda a: a[~idx], (y,) + args)
    priv = map(lambda a: a[idx], (y,) + args)
    if sample_weight is not None:
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)