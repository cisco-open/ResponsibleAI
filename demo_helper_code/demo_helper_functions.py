import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
__all__ = ["get_german_dataset", "reweigh_dataset_for_age", "Net", "convertSklearnToTensor", "convertSklearnToDataloader"]


default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Old', 0.0: 'Young'}],
}


def default_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['sex'].replace(status_map)
    return df


def get_german_dataset():
    favorable_classes = [1]
    protected_attribute_names = ['sex']
    privileged_classes = ['male']
    categorical_features = ['status', 'credit_history', 'purpose',
         'savings', 'employment', 'sex', 'other_debtors', 'property', 'age',
         'installment_plans', 'housing', 'skill_level', 'telephone',
         'foreign_worker']

    categorical_meanings = {
        "status": {0: "< 0 DM", 1: "0<= ... < 200 DM", 2: " >= 200 DM", 3: "No checking account"},
        "credit_history": {0: "No credits taken", 1: "All credits paid back duly", 2: "Existing credits paid until now", 3: "Delay in past payments", 4: "Critical Account"},
        "purpose": {0: "New car", 1: "Used car", 2: "Furniture", 3: "Radio/Television", 4: "Domestic Appliance", 5:"Repairs", 6:"Education", 7:"Vacation", 8:"Retraining", 9:"Business", 10:"Other"},
        "savings": {0: "<100 DM", 1:"< 500 DM", 2: "<1000 DM", 3:">= 1000 DM", 4:"Unknown"},
        "employment": {0:"Unemployed", 1:"<1 year", 2:"<4 years", 3:"<7 years", 4:">=7 Years"},
        'sex': {0:'female', 1:'male'},
        'other_debtors': {0:"none", 1:'co-applicant', 2:'guarantor'},
        'age': {0: '<=25', 1: '>25'},
        'property': {0:'real estate', 1:'life insurance', 2:'car or other', 3:'no property'},
        'installment_plans': {0:'bank', 1:'stores', 2:'none'},
        'housing': {0:'rent', 1:'own', 2:'for free'},
        'job': {0:'unemployed', 1:'resident', 2:'official', 3:'management'},
        'telephone': {0:'none', 1:'yes'},
        'foreign_worker': {0:'yes', 1:'no'}}

    na_values = []
    custom_preprocessing = default_preprocessing
    metadata = default_mappings

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demo_helper_code', 'datasets', 'german.data')
    # as given by german.doc
    column_names = ['status', 'month', 'credit_history',
        'purpose', 'credit_amount', 'savings', 'employment',
        'investment_as_income_percentage', 'sex',
        'other_debtors', 'residence_since', 'property', 'age',
        'installment_plans', 'housing', 'number_of_credits',
        'skill_level', 'people_liable_for', 'telephone',
        'foreign_worker', 'credit']
    try:
        df = pd.read_csv(filepath, sep=' ', header=None, names=column_names, na_values=na_values)
        df = default_preprocessing(df)
        for name in column_names:
            if name in categorical_features:
                df[name] = pd.factorize(df[name], sort=True)[0]
        df['credit'] = df['credit'].apply(lambda x: 0 if x == 2 else 1)
        df['age'] = df['age'].apply(lambda x: 0 if x <= 25 else 1)

    except IOError as err:
        print("IOError: {}".format(err))
        print("To use this class, please download the following files:")
        print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
        print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc")
        print("\nand place them, as-is, in the folder:")
        print("\n\t{}\n".format(os.path.abspath(os.path.join(
            os.path.abspath(__file__), '..', '..', 'data', 'raw', 'german'))))
        import sys
        sys.exit(1)

    return {"df": df, "protected_attribute_names": ["age"], "privileged_info": {"age": {"privileged": 0, "unprivileged": 1}},
            "categorical_meanings": categorical_meanings, "positive_label": 1}


def reweigh_dataset_for_age(df, y):
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import StandardDataset
    df['credit'] = y
    label_name = 'credit'
    favorable_classes = [1]
    protected_attribute_names = ['age']
    privileged_classes = [[0]]
    metadata={'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Old', 0.0: 'Young'}]}
    my_test = StandardDataset(df, label_name, favorable_classes, protected_attribute_names, privileged_classes, metadata=metadata)
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(my_test)
    new_df = dataset_transf_train.convert_to_dataframe()[0]

    from sklearn.model_selection import train_test_split
    y = new_df.pop("credit")
    X = new_df
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)
    xTrain = xTrain.to_numpy()
    xTest = xTest.to_numpy()
    yTrain = yTrain.to_numpy()
    yTest = yTest.to_numpy()
    return xTrain, xTest, yTrain, yTest



class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 300).to("cpu")
        self.fc2 = nn.Linear(300, 200).to("cpu")
        self.fc3 = nn.Linear(200, 80).to("cpu")
        self.fc4 = nn.Linear(80, 2).to("cpu")

    def forward(self, x):
        x = F.relu(self.fc1(x)).to("cpu")
        x = F.relu(self.fc2(x)).to("cpu")
        x = F.relu(self.fc3(x)).to("cpu")
        x = self.fc4(x).to("cpu")
        return x.to("cpu")


def convertSklearnToTensor(xTrain, xTest, yTrain, yTest):
    import torch
    n_values = np.max(yTest) + 1
    yTrain_1h = np.eye(n_values)[
        yTrain]  # 1-hot representation of output classes, to match the criteria of the loss function.
    yTest_1h = np.eye(n_values)[
        yTest]  # 1-hot representation of output classes, to match the criteria of the loss function.

    # Convert sklearn dataset to pytorch's format.
    X_train_t = torch.from_numpy(xTrain).to(torch.float32).to("cpu")
    y_train_t = torch.from_numpy(yTrain_1h).to(torch.float32).to("cpu")
    X_test_t = torch.from_numpy(xTest).to(torch.float32).to("cpu")
    y_test_t = torch.from_numpy(yTest_1h).to(torch.float32).to("cpu")
    return X_train_t, y_train_t, X_test_t, y_test_t



def convertSklearnToDataloader(xTrain, xTest, yTrain, yTest):
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    X_train_t, y_train_t, X_test_t, y_test_t = convertSklearnToTensor(xTrain, xTest, yTrain, yTest)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_dataloader = DataLoader(train_dataset, batch_size=150)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_dataloader = DataLoader(test_dataset, batch_size=150)
    return train_dataloader, test_dataloader