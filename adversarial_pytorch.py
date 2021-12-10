# Training not importnat just call function
# Hide all training, give function to get pytorch models.
# Use a closer scale.


# Setup environment for pytorch
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"


# Create instance of pytorch network
from demo_helper_code.demo_helper_functions import Net
net = Net(input_size=30, scale=10).to("cpu")
net2 = Net(input_size=30, scale=2).to("cpu")
# Repeat with different scales


# Get Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x, y = load_breast_cancer(return_X_y=True)
xTrain, xTest, yTrain, yTest = train_test_split(x, y)


# Scale data
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)


# Convert Sklearn data to a Pytorch Tensor representation so the Net can run on them.
from demo_helper_code.demo_helper_functions import convertSklearnToDataloader
train_dataloader, test_dataloader = convertSklearnToDataloader(xTrain, xTest, yTrain, yTest)


# Define Pytorch Optimizer and Loss Function
criterion = nn.CrossEntropyLoss().to("cpu")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)


# Very basic pytorch training cycle
for epoch in range(300):
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print('Finished Training')


# Very basic pytorch training cycle
for epoch in range(300):
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net2(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print('Finished Training')



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
model = Model(agent=net, name="cisco_cancer_pytorch_robustness", display_name="Cisco Pytorch Robustness", model_class="Neural Network", adaptive=True,
              optimizer=optimizer, loss_function=criterion)
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {"time_complexity": "polynomial"}
ai_pytorch = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration, custom_certificate_location="RAI\\certificates\\standard\\cert_list_ad_demo_ptc.json")
ai_pytorch.initialize()


# Create predictions from the training data
# Put the data in tensor format and make predictions.
from demo_helper_code.demo_helper_functions import convertSklearnToTensor
X_train_t, y_train_t, X_test_t, y_test_t = convertSklearnToTensor(xTrain, xTest, yTrain, yTest)
train_preds = torch.argmax(net(X_train_t), axis=1)


# Compute and store metrics about pytorch predictions.
ai_pytorch.reset_redis()
ai_pytorch.compute_metrics(train_preds.cpu(), data_type="train")
ai_pytorch.export_data_flat("Pytorch Model")
ai_pytorch.compute_certificates()
ai_pytorch.export_certificates("Neural Net")



train_preds = torch.argmax(net2(X_train_t), axis=1)
ai_pytorch.task.model.agent = net2
ai_pytorch.compute_metrics(train_preds.cpu(), data_type="train")
ai_pytorch.export_data_flat("Pytorch Model2")
ai_pytorch.compute_certificates()
ai_pytorch.export_certificates("NN 2")



# View GUI
ai_pytorch.viewGUI()

