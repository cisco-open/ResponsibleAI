# Define Pytorch Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# FIRST DRAFT, CODE SOON TO CLEANED UP, COMPLETED AND PUT IN JUPYTER.
# MULTIPLE WEBSITE INSTANCES NEEDED TO BE ADDED TO RAI FIRST TO ALLOW FOR THE DEMO.

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 80)
        self.fc4 = nn.Linear(80, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net().to("cpu")

# Get Dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x, y = load_breast_cancer(return_X_y=True)

xTrain, xTest, yTrain, yTest = train_test_split(x, y)
n_values = np.max(yTest) + 1
yTrain_1h = np.eye(n_values)[yTrain]

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)

# Convert data to pytorch:
X_train_t = torch.from_numpy(xTrain).to(torch.float32).to("cpu")
y_train_t = torch.from_numpy(yTrain_1h).to(torch.float32).to("cpu")
train_dataset = TensorDataset(X_train_t, y_train_t)
train_dataloader = DataLoader(train_dataset, batch_size=150)


# Define Optimizer and Loss Function
criterion = nn.CrossEntropyLoss().to("cpu")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
optimizer


# Train Loop
for epoch in range(300):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 and i == len(train_dataloader)-1:  # print every 2000 mini-batches
            outputs = torch.argmax(outputs, axis=1)
            labels = torch.argmax(labels, axis=1)
            correct = (outputs == labels).float().sum() / len(outputs)

print('Finished Training')

# ADD RAI
from RAI.dataset import Feature, Data, MetaDatabase, Dataset
from RAI.AISystem import AISystem, Model, Task

# Put features in RAI format.
features_raw = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "diagnosis"]
features = []
for feature in features_raw:
    features.append(Feature(feature, "float32", feature))


# Hook data in with our Representation
training_data = Data(xTrain, yTrain)  # Accepts Data and GT
test_data = Data(xTest, yTest)
dataset = Dataset(training_data, test_data=test_data)  # Accepts Training, Test and Validation Set
meta = MetaDatabase(features)


model = Model(agent=net, name="cisco_cancer_ai", display_name="Cisco Health AI", model_class="Neural Network", adaptive=True,
              optimizer=optimizer, loss_function=criterion)
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {"time_complexity": "polynomial"}
ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
ai.initialize()


train_preds = torch.argmax(net(X_train_t), axis=1)

ai.reset_redis()
ai.compute_metrics(train_preds.cpu(), data_type="train")
ai.export_data_flat("Pytorch Model")
ai.export_certificates()



from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
model = Model(agent=reg, name="cisco_cancer_ai_2", display_name="Cisco Health AI Forest", model_class="Random Forest Classifier", adaptive=False)
# Indicate the task of the model
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {}

ai = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration)
ai.initialize()

# Train model
reg.fit(xTrain, yTrain)
train_preds = reg.predict(xTrain)

# Make Predictions
ai.reset_redis()
# ai.compute_metrics(train_preds, data_type="train", export_title="Train set")

test_preds = reg.predict(xTest)
# Make Predictions

ai.compute_metrics(test_preds, data_type="test", export_title="Test set")
ai.export_data_flat("Sklearn Model")
ai.export_certificates()






