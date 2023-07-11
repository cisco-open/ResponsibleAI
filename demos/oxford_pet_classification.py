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
# This demo uses oxford pet dataset and shows how RAI can be used to evaluate image processing tasks during training


# importing modules
import os
import sys
import inspect
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from dotenv import load_dotenv
from torchvision.models import regnet_y_800mf

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.db.service import RaiDB
from RAI.dataset import MetaDatabase, Feature, Dataset, IteratorData

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
load_dotenv(f'{currentdir}/../.env')


def main():
    # Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(10)
    PATH = './oxford_pet_net.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get Data
    batch_size = 128
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.4484, 0.3949), (0.2693, 0.2648, 0.2728)),
                                          transforms.Resize(256), transforms.CenterCrop(224)])
    train_set = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval", download=True,
                                                   transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=True,
                                                  transform=train_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    output_mapping = train_set.class_to_idx

    # Get a Finetuned model for transfer learning
    net = regnet_y_800mf(pretrained=True)
    net.fc = nn.Sequential(torch.nn.Linear(784, 1000), torch.nn.Linear(1000, len(output_mapping)))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Define predict function to return class with highest value
    def predict(input_image):
        if not isinstance(input_image, torch.Tensor):
            input_image = torch.Tensor(input_image)
        _, predicted = torch.max(net(input_image), 1)
        return predicted.tolist()

    # Define the output of the model and the RAI model
    output_mapping = {int(v): k for k, v in output_mapping.items()}
    outputs = Feature('image_type', 'Numeric', 'The type of image', categorical=True, values=output_mapping)
    model = Model(agent=net, output_features=outputs, name="reg_net", predict_fun=predict, description="RegNet",
                  model_class="RegNet", loss_function=criterion, optimizer=optimizer)

    # Train the model
    def train():
        for name, param in net.named_parameters():
            if name not in ["fc.weight", "fc.bias"] and not 'block4' in name and not 'block3' in name:
                param.requires_grad = False

        print("Starting training")
        for epoch in range(90):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 20 == 19:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0
            torch.save(net.state_dict(), PATH)
            scheduler.step()

        # Freeze all but last layer
        for name, param in net.named_parameters():
            param.requires_grad = True
        torch.save(net.state_dict(), PATH)

    # Load the model if it exists, otherwise train one
    if os.path.isfile(PATH):
        print("Loading model")
        net.load_state_dict(torch.load(PATH, map_location=device))
    else:
        train()

    # Define model parameters required for Gradcam: f1, classifier, features_conv, flatten
    net.f1 = nn.Sequential(net.avgpool)
    net.classifier = net.fc
    net.features_conv = nn.Sequential(net.stem, net.trunk_output)
    net.flatten = True

    # Define the RAI input and output features
    image = Feature('image', 'Image', 'The 32x32 input image')
    meta = MetaDatabase([image])

    # Pass test data split to RAI
    dataset = Dataset({"test": IteratorData(test_loader)})

    # Create the RAI AISystem
    configuration = {"time_complexity": "polynomial"}
    ai = AISystem(name="oxford_pets_class", task='classification', meta_database=meta, dataset=dataset, model=model)
    ai.initialize(user_config=configuration)

    preds = []
    net.eval()
    print("Generating predictions")
    with torch.no_grad():
        for i, vals in enumerate(test_loader, 0):
            if i % int(len(test_loader) / 20) == 0:
                print(str(int(100 * i / len(test_loader))), "% Done")
            image, label = vals
            _, predicted = torch.max(net(image), 1)
            preds += predicted.tolist()

    # Compute Metrics based on the predictions
    ai.compute({"test": {"predict": preds}}, tag='regnet')

    # View the dashboard
    net.eval()
    r = RaiDB(ai)
    r.reset_data()
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")


if __name__ == '__main__':
    main()
