# Set up OS environment to only use desired devices.
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"


# Create instance of pytorch network
from demo_helper_code.demo_helper_functions import Net
net = Net(30).to("cpu")


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


# Converts Sklearn data to a Pytorch Tensor representation so the Net can run on them.
from demo_helper_code.demo_helper_functions import convertSklearnToDataloader
train_dataloader, test_dataloader = convertSklearnToDataloader(xTrain, xTest, yTrain, yTest)


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
model = Model(agent=net, name="cisco_ai_train_cycle", display_name="Cisco AI Train Test", model_class="Neural Network", adaptive=True,
              optimizer=optimizer, loss_function=criterion)
task = Task(model=model, type='binary_classification', description="Detect Cancer in patients using skin measurements")
configuration = {"time_complexity": "polynomial"}
ai_pytorch = AISystem(meta_database=meta, dataset=dataset, task=task, user_config=configuration, custom_certificate_location="RAI\\certificates\\standard\\cert_list_ad_demo_ptc.json")
ai_pytorch.initialize()


# Compute and store metrics about pytorch predictions.
ai_pytorch.reset_redis()


# Train the model
def train(train_dataloader):
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to("cpu")
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Test the model
def test(epoch, test_dataloader):
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = net(inputs).to("cpu")

    # Compute metrics on the outputs of the test metrics
    outputs = torch.argmax(outputs, axis=1)
    ai_pytorch.compute_metrics(outputs.to("cpu"), data_type="test")
    ai_pytorch.export_data_flat(str(epoch))
    ai_pytorch.compute_certificates()
    ai_pytorch.export_certificates(str(epoch))


# Run train test cycle
for epoch in range(100):
    train(train_dataloader)
    if epoch%10 == 0:
        test(epoch, test_dataloader)


# View metrics and certificates to determine the best model to use.
ai_pytorch.viewGUI()
