import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import random


__all__ = ["get_german_dataset", "reweigh_dataset_for_age", "Net", "convertSklearnToTensor", "convertSklearnToDataloader",
           "get_rai_dataset", "get_rai_metadatabase", 'get_classifier_and_preds', 'get_german_rai_ai_system', 'get_untrained_net',
           'load_breast_cancer_dataset', 'get_breast_cancer_metadatabase', 'get_breast_cancer_rai_ai_system', 'train_net',
           'get_net_test_preds', 'get_ai_trees', 'get_trained_net']


# GERMAN DATASET VALUES
default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Old', 0.0: 'Young'}],
}

# Default processing for the german dataset
def default_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['sex'].replace(status_map)
    return df


# Get the german credit dataset
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

    # Put data in format where predictions can be made.
    from sklearn.model_selection import train_test_split
    y = df.pop("credit")
    X = df
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=2, stratify=y)
    xTrain = xTrain.to_numpy()
    xTest = xTest.to_numpy()
    yTrain = yTrain.to_numpy()
    yTest = yTest.to_numpy()

    df_info = {"X": X, "protected_attribute_names": ["age"], "privileged_info": {"age": {"privileged": 0, "unprivileged": 1}},
            "categorical_meanings": categorical_meanings, "positive_label": 1}

    return df_info, X, y, xTrain, xTest, yTrain, yTest


# Use the reweighting algorithm on the german dataset DF to potentially solve bias issues
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


# Standard fully connected neural network for very basic predictions
class Net(nn.Module):
    def __init__(self, input_size=30, scale=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 30*scale).to("cpu")
        self.fc2 = nn.Linear(30*scale, 20*scale).to("cpu")
        self.fc3 = nn.Linear(20*scale, 8*scale).to("cpu")
        self.fc4 = nn.Linear(8*scale, 2).to("cpu")

    def forward(self, x):
        x = F.relu(self.fc1(x)).to("cpu")
        x = F.relu(self.fc2(x)).to("cpu")
        x = F.relu(self.fc3(x)).to("cpu")
        x = self.fc4(x).to("cpu")
        return x.to("cpu")


# Converts a dataset list from Sklearn to a Pytorch tensors.
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


# Converts Sklearn datasets to Pytorch dataloaders.
def convertSklearnToDataloader(xTrain, xTest, yTrain, yTest):
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    X_train_t, y_train_t, X_test_t, y_test_t = convertSklearnToTensor(xTrain, xTest, yTrain, yTest)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_dataloader = DataLoader(train_dataset, batch_size=150, shuffle=False)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=150)
    return train_dataloader, test_dataloader


def get_rai_dataset(xTrain, xTest, yTrain, yTest):
    from RAI.dataset import Dataset, Data
    training_data = Data(xTrain, yTrain)  # Accepts Data and GT
    test_data = Data(xTest, yTest)
    return Dataset(training_data, test_data=test_data)  # Accepts Training, Test and Validation Set


def get_rai_metadatabase(df_info):
    from RAI.utils import df_to_meta_database
    meta, fairness_config = df_to_meta_database(df_info['X'], categorical_values=df_info["categorical_meanings"],
                                                protected_attribute_names=df_info["protected_attribute_names"],
                                                privileged_info=df_info["privileged_info"],
                                                positive_label=df_info["positive_label"])
    return meta, fairness_config


def get_classifier_and_preds(xTrain, xTest, yTrain, reg=None):
    from sklearn.ensemble import RandomForestClassifier
    if reg == None:
        reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    reg.fit(xTrain, yTrain)
    train_preds = reg.predict(xTrain)
    test_preds = reg.predict(xTest)
    return reg, train_preds, test_preds


def load_breast_cancer_dataset(pytorch=False):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    x, y = load_breast_cancer(return_X_y=True)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.fit_transform(xTest)
    if(pytorch):
        train_dataloader, test_dataloader = convertSklearnToDataloader(xTrain, xTest, yTrain, yTest)
        return train_dataloader, test_dataloader, xTrain, xTest, yTrain, yTest
    return xTrain, xTest, yTrain, yTest


def get_german_rai_ai_system(reg, rai_fairness_config, rai_MetaDatabase, rai_dataset):
    from RAI.AISystem import AISystem, Model, Task
    model = Model(agent=reg, name="cisco_german_fairness", display_name="Cisco German Fairness",
                  model_class="Random Forest Classifier", adaptive=False)
    task = Task(model=model, type='binary_classification', description="Predict the credit score of various Germans.")
    configuration = {"fairness": rai_fairness_config, "time_complexity": "linear"}
    credit_ai = AISystem(meta_database=rai_MetaDatabase, dataset=rai_dataset, task=task, user_config=configuration,
                         custom_certificate_location="cert_list_credit.json")
    credit_ai.initialize()
    return credit_ai


# Train Test
def get_breast_cancer_rai_ai_system(net, optimizer, criterion, rai_MetaDatabase, rai_dataset, cert_loc="cert_list_ad_demo_ptc.json"):
    from RAI.AISystem import AISystem, Model, Task
    model = Model(agent=net, name="cisco_ai_train_cycle", display_name="Cisco AI Train Test",
                  model_class="Neural Network", adaptive=True,
                  optimizer=optimizer, loss_function=criterion)
    task = Task(model=model, type='binary_classification',
                description="Detect Cancer in patients using skin measurements")
    configuration = {"time_complexity": "polynomial"}
    ai_pytorch = AISystem(meta_database=rai_MetaDatabase, dataset=rai_dataset, task=task, user_config=configuration,
                          custom_certificate_location="RAI\\certificates\\standard\\" + cert_loc)
    ai_pytorch.initialize()
    ai_pytorch.reset_redis()
    return ai_pytorch


def get_untrained_net(input_size=30, scale=10):
    import torch
    import torch.nn as nn
    net = Net(input_size=input_size, scale=scale).to("cpu")
    criterion = nn.CrossEntropyLoss().to("cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
    return net, criterion, optimizer


def get_trained_net(input_size=30, scale=10, train_dataloader=None, epochs=100):
    import torch
    import torch.nn as nn
    net = Net(input_size=input_size, scale=scale).to("cpu")
    criterion = nn.CrossEntropyLoss().to("cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
    for epoch in range(epochs):
        train_net(net, optimizer, criterion, train_dataloader)
    return net, criterion, optimizer


def get_breast_cancer_metadatabase():
    from RAI.dataset import Feature, MetaDatabase
    features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                    "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                    "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                    "texture_worst", "perimeter_worst", "area_worst",
                    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                    "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
    features = []
    for feature in features_raw:
        features.append(Feature(feature, "float32", feature))
    return MetaDatabase(features)



# Train the model
def train_net(net, optimizer, criterion, train_dataloader):
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to("cpu")
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def get_net_test_preds(net, test_dataloader):
    import torch
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = net(inputs).to("cpu")
    # Compute metrics on the outputs of the test metrics
    return torch.argmax(outputs, axis=1)



# Test the model
def test(ai_pytorch, net, epoch, test_dataloader):
    outputs = get_net_test_preds(net, test_dataloader)
    ai_pytorch.compute_metrics(outputs.to("cpu"), data_type="test", export_title=(str(epoch)))


def get_ai_trees(xTrain, yTrain):
    from sklearn.ensemble import RandomForestClassifier
    reg_rf = RandomForestClassifier(n_estimators=10, max_depth=10, criterion='entropy', random_state=0)
    reg_rf.fit(xTrain, yTrain)

    reg_dt = RandomForestClassifier(n_estimators=1, max_depth=10, random_state=0)
    reg_dt.fit(xTrain, yTrain)

    from sklearn.ensemble import GradientBoostingClassifier
    reg_gb = GradientBoostingClassifier(n_estimators=3, max_depth=10, random_state=0)
    reg_gb.fit(xTrain, yTrain)

    return reg_rf, reg_dt, reg_gb

