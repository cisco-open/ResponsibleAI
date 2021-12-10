# Define Pytorch Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 300).to("cpu")
        self.fc2 = nn.Linear(300, 200).to("cpu")
        self.fc3 = nn.Linear(200, 80).to("cpu")
        self.fc4 = nn.Linear(80, 2).to("cpu")

    def forward(self, x):
        x = F.relu(self.fc1(x)).to("cpu")
        x = F.relu(self.fc2(x)).to("cpu")
        x = F.relu(self.fc3(x)).to("cpu")
        x = self.fc4(x).to("cpu")
        return x.to("cpu")


# Create instance of pytorch network
net = Net().to("cpu")


# Get Dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x, y = load_breast_cancer(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)
n_values = np.max(yTest) + 1
yTrain_1h = np.eye(n_values)[yTrain]  # 1-hot representation of output classes, to match the criteria of the loss function.
yTest_1h = np.eye(n_values)[yTest]  # 1-hot representation of output classes, to match the criteria of the loss function.


# Scale data
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)


# Convert sklearn dataset to pytorch's format.
X_train_t = torch.from_numpy(xTrain).to(torch.float32).to("cpu")
y_train_t = torch.from_numpy(yTrain_1h).to(torch.float32).to("cpu")
X_test_t = torch.from_numpy(xTest).to(torch.float32).to("cpu")
y_test_t = torch.from_numpy(yTest_1h).to(torch.float32).to("cpu")


train_dataset = TensorDataset(X_train_t, y_train_t)
train_dataloader = DataLoader(train_dataset, batch_size=150)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_dataloader = DataLoader(test_dataset, batch_size=150)


# Define Pytorch Optimizer and Loss Function
criterion = nn.CrossEntropyLoss().to("cpu")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)



# Convert current dataset to RAI's representation.
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
features = []
for feature in features_raw:
    features.append(Feature(feature, "float32", feature))


# Create RAI Data objects for the datasets.
training_data = Data(xTrain, yTrain)  # Accepts Data and GT
test_data = Data(xTest, yTest)
dataset = Dataset(training_data, test_data=test_data)  # Accepts Training, Test and Validation Set
meta = MetaDatabase(features)


# Create RAI model and set the agent as the pytorch network.
model = Model(agent=net, name="cisco_cancer_ai_train_cycle", display_name="Cisco Health AI Pytorch", model_class="Neural Network", adaptive=True,
              optimizer=optimizer, loss_function=criterion)
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {"time_complexity": "polynomial"}
ai_pytorch = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration, custom_certificate_location="RAI\\certificates\\standard\\cert_list_ad_demo_ptc.json")
ai_pytorch.initialize()


# Compute and store metrics about pytorch predictions.
ai_pytorch.reset_redis()

# train the model
def train(train_dataloader):
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to("cpu")
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# evaluate the model
def test(epoch, test_dataloader):
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = net(inputs).to("cpu")
    outputs = torch.argmax(outputs, axis=1)
    ai_pytorch.compute_metrics(outputs.to("cpu"), data_type="test")
    ai_pytorch.export_data_flat(str(epoch))
    ai_pytorch.compute_certificates()
    ai_pytorch.export_certificates(str(epoch))


for epoch in range(100):
    train(train_dataloader)
    if epoch%10 == 0:
        test(epoch, test_dataloader)


