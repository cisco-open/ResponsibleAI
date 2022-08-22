__all__ = ["get_binary_dataset", "get_classification_dataset"]
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
import pandas as pd


def get_binary_dataset(metric_group, data, prot_attr):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features if feature.categorical]
    df = pd.DataFrame(data.categorical, columns=names)
    df['y'] = data.y
    bin_dataset = BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=prot_attr)
    return BinaryLabelDatasetMetric(bin_dataset)


def get_classification_dataset(metric_group, data, preds, prot_attr, priv_group_list, unpriv_group_list):
    names = [feature.name for feature in metric_group.ai_system.meta_database.features if feature.categorical]
    df1 = pd.DataFrame(data.categorical, columns=names)
    df1['y'] = data.y
    df2 = pd.DataFrame(data.categorical, columns=names)
    df2['y'] = preds
    binDataset1 = BinaryLabelDataset(df=df1, label_names=['y'], protected_attribute_names=prot_attr)
    binDataset2 = BinaryLabelDataset(df=df2, label_names=['y'], protected_attribute_names=prot_attr)
    return ClassificationMetric(binDataset1, binDataset2, unprivileged_groups=unpriv_group_list, privileged_groups=priv_group_list)
