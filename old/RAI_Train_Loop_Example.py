from demo_helper_code.demo_helper_functions import *
import os
import random
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
torch.manual_seed(1)
random.seed(5)
np.random.seed(10)
# 2 minutes


# Create instance of pytorch network
net, criterion, optimizer = get_untrained_net(input_size=30, scale=10)


# Get Breast cancer data, with pytorch representation
train_dataloader, test_dataloader, xTrain, xTest, yTrain, yTest = load_breast_cancer_dataset(pytorch=True)


# Get RAI data representations of the breast cancer dataset.
rai_MetaDatabase = get_breast_cancer_metadatabase()
rai_dataset = get_rai_dataset(xTrain, xTest, yTrain, yTest)


# !! Get RAI AI system for the dataset.
ai_pytorch = get_breast_cancer_rai_ai_system(net, optimizer, criterion, rai_MetaDatabase, rai_dataset, cert_loc="cert_list_ad_demo_ptc.json")


# View metrics and certificates to determine the best model to use.
ai_pytorch.viewGUI()


# Test the model
def test(ai_pytorch, net, epoch, test_dataloader):
    outputs = get_net_test_preds(net, test_dataloader)
    ai_pytorch.compute_metrics(outputs.to("cpu"), data_type="test", export_title=(str(epoch)))


def run_train_test_cycle(ai_pytorch, net, optimizer, criterion, train_dataloader, test_dataloader, epochs=100):
    for epoch in range(epochs):
        train_net(net, optimizer, criterion, train_dataloader)
        if epoch % 10 == 0:
            test(ai_pytorch, net, epoch, test_dataloader)


# Run train test cycle
run_train_test_cycle(ai_pytorch, net, optimizer, criterion, train_dataloader, test_dataloader, epochs=101)

