from demo_helper_code.demo_helper_functions import *
import torch
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
torch.manual_seed(0)
random.seed(0)


# Get Breast cancer data, with pytorch representation
train_dataloader, test_dataloader, xTrain, xTest, yTrain, yTest = load_breast_cancer_dataset(pytorch=True)


# Train pytorch neural networks of different sizes on the data.
net1, criterion1, optimizer1 = get_trained_net(input_size=30, scale=5, train_dataloader=train_dataloader, epochs=100)
net2, criterion2, optimizer2 = get_trained_net(input_size=30, scale=10, train_dataloader=train_dataloader, epochs=100)
net3, criterion3, optimizer3 = get_trained_net(input_size=30, scale=20, train_dataloader=train_dataloader, epochs=100)


# Get RAI data representations of the breast cancer dataset.
rai_MetaDatabase = get_breast_cancer_metadatabase()
rai_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)


# Initialize the RAI AI system for this network.
ai_pytorch = get_breast_cancer_rai_ai_system(net1, optimizer1, criterion1, rai_MetaDatabase, rai_dataset, cert_loc="cert_list_ad_demo_ptc.json")


# Get Neural Network Predictions on Data for each Network.
X_train_t, y_train_t, X_test_t, y_test_t = convertSklearnToTensor(xTrain, xTest, yTrain, yTest)
test_preds_1 = torch.argmax(net1(X_test_t), axis=1)
test_preds_2 = torch.argmax(net2(X_test_t), axis=1)
test_preds_3 = torch.argmax(net3(X_test_t), axis=1)


# Compute and store metrics about pytorch predictions.
ai_pytorch.compute_metrics(test_preds_1.cpu(), data_type="test", export_title="Scale 5")

# Replace the AI System parts with elements of Net2.
ai_pytorch.task.model.agent = net2
ai_pytorch.task.model.optimizer = optimizer2
ai_pytorch.task.model.loss_function = criterion2

# Recompute metrics.
ai_pytorch.compute_metrics(test_preds_2.cpu(), data_type="test", export_title="Scale 10")


# Replace the AI System parts with elements of Net3.
ai_pytorch.task.model.agent = net3
ai_pytorch.task.model.optimizer = optimizer3
ai_pytorch.task.model.loss_function = criterion3

# Recompute metrics.
ai_pytorch.compute_metrics(test_preds_3.cpu(), data_type="test", export_title="Scale 20")


# View GUI
ai_pytorch.viewGUI()

