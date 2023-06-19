# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Description
# This demo uses Cifar10 dataset and shows how RAI can be used monitor image processing tasks during training


# importing modules
import os
import sys
import inspect
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from dotenv import load_dotenv

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.db.service import RaiDB
from RAI.utils import torch_to_RAI
from RAI.dataset import MetaDatabase, Feature, Dataset, NumpyData

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

load_dotenv(f'{currentdir}/../.env')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(10)

    # Get Torch Data
    batch_size = 4
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Convert train data to RAIs representation
    x_test_data, y_test_data, raw_x_test_data = torch_to_RAI(test_loader, detailed=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define net
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features_conv = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
            )
            self.f1 = nn.Sequential(
                nn.MaxPool2d(2, 2),
            )
            self.flatten = True  # True if flatten is needed for fc
            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )

        def forward(self, x):
            x = self.features_conv(x)
            x = self.f1(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Create train and test functions
    def train(epoch):
        print("Training Model")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[Epoch: {epoch + 1}, Mini-Batch: {i + 1:5d}] Loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    def test(epoch: int, ai, rai):
        print("Testing Model")
        preds = []
        for i, vals in enumerate(test_loader, 0):
            image, label = vals
            _, predicted = torch.max(net(image), 1)
            preds += predicted
        # Pass test set predictions to RAI
        ai.compute({"test": {"predict": preds}}, tag='Epoch: ' + str(epoch))
        rai.add_measurement()
        rai.export_metadata()

    # Train and test loop
    def train_test(conv_net, rai):
        for epoch in range(20):
            train(epoch)
            test(epoch, conv_net, rai)

    # The predict function to pass to RAI
    def predict(input_image):
        if not isinstance(input_image, torch.Tensor):
            input_image = torch.Tensor(input_image)
        return net(input_image)

    # Define input and output features of the model
    image = Feature('Input Image', 'image', 'The 32x32 input image')
    outputs = Feature('Image Class', 'numeric', 'The type of image', categorical=True, values={i: v for i, v in enumerate(classes)})

    # Create the meta database describing the dataset
    meta = MetaDatabase([image])
    # Create a RAI Model
    model = Model(agent=net, output_features=outputs, name="conv_net", predict_fun=predict, description="ConvNet", model_class="ConvNet",
                  loss_function=criterion, optimizer=optimizer)
    # Convert the data split to a RAI Dataset
    dataset = Dataset({"test": NumpyData(x_test_data, y_test_data, raw_x_test_data)})

    # Create a RAI AI System
    ai = AISystem(name="cifar_classification_train", task='classification', meta_database=meta, dataset=dataset, model=model)
    configuration = {"time_complexity": "polynomial"}
    ai.initialize(user_config=configuration)

    # Turn on the dashboard
    r = RaiDB(ai)
    r.reset_data()

    # Run the training and test loop
    train_test(ai, r)
    r.export_visualizations("test", "test")


if __name__ == '__main__':
    main()
