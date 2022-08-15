import sys
import inspect
from torch.optim.lr_scheduler import StepLR
from RAI.AISystem import AISystem, Model
from RAI.redis import RaiRedis
from RAI.utils import torch_to_RAI
from RAI.dataset import MetaDatabase, Feature, Dataset, Data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from torchvision.models import resnet50, regnet_y_800mf
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(10)
    PATH = './oxford_pet_net.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 128
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.4484, 0.3949), (0.2693, 0.2648, 0.2728)),
         transforms.Resize(256), transforms.CenterCrop(224)])
    train_set = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval", download=True,
                                                   transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=True,
                                                  transform=train_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    output_mapping = train_set.class_to_idx

    # Get a finetuned model
    net = regnet_y_800mf(pretrained=True)
    net.fc = nn.Sequential(torch.nn.Linear(784, 1000), torch.nn.Linear(1000, len(output_mapping)))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Define predict function to use for RAI
    def predict(input_image):
        if not isinstance(input_image, torch.Tensor):
            input_image = torch.Tensor(input_image)
        _, predicted = torch.max(net(input_image), 1)
        return predicted.tolist()

    # Pass model to RAI
    output_mapping = {int(v): k for k, v in output_mapping.items()}
    outputs = Feature('image_type', 'Numeric', 'The type of image', categorical=True, values=output_mapping)
    model = Model(agent=net, output_features=outputs, name="reg_net", predict_fun=predict, description="RegNet",
                  model_class="RegNet", loss_function=criterion, optimizer=optimizer)

    # Train the model
    def train():
        # Freeze all but last layer
        for name, param in net.named_parameters():
            if name not in ["fc.weight", "fc.bias"] and not 'block4' in name and not 'block3' in name:
                param.requires_grad = False

        print("Starting training")
        for epoch in range(90):  # loop over the dataset multiple times
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


    def test():
        net.eval()
        with torch.no_grad():
            n_val_correct = 0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)

                n_val_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        val_acc = 100. * n_val_correct / len(test_loader.dataset)
        print("Accuracy: ", val_acc)

    # Load the model if it exists, otherwise train one
    if os.path.isfile(PATH):
        print("Loading model")
        net.load_state_dict(torch.load(PATH, map_location=device))
    else:
        train()

    # test()

    # Define model parameters required for Gradcam: f1, classifier, features_conv, flatten
    net.f1 = nn.Sequential(net.avgpool)
    net.classifier = net.fc
    net.features_conv = nn.Sequential(net.stem, net.trunk_output)
    net.flatten = True

    # Get the data in RAIs format
    print("putting it in RAI format")
    x_test_data, y_test_data, raw_x_test_data = torch_to_RAI(test_loader)

    # Define the RAI input and output features
    image = Feature('image', 'Image', 'The 32x32 input image')
    meta = MetaDatabase([image])

    # Pass test data split to RAI
    dataset = Dataset({"test": Data(x_test_data, y_test_data, raw_x_test_data)})

    # Create the RAI AISystem
    interpret_method = []  # ["gradcam"]
    configuration = {"time_complexity": "polynomial"}
    ai = AISystem(name="oxford_pets_class", task='classification', meta_database=meta, dataset=dataset, model=model,
                  interpret_methods=interpret_method)
    ai.initialize(user_config=configuration)



    # Generate predictions
    import pickle
    preds = []
    file_name = "saved_preds.pkl"

    # Temp measure to run code faster
    if os.path.isfile(file_name):
        print("Loading preds")
        open_file = open(file_name, "rb")
        preds = pickle.load(open_file)
        open_file.close()
    else:
        net.eval()
        print("Generating predictions")
        with torch.no_grad():
            for i, vals in enumerate(test_loader, 0):
                if i % int(len(test_loader) / 20) == 0:
                    print(str(int(100 * i / len(test_loader))), "% Done")
                image, label = vals
                _, predicted = torch.max(net(image), 1)
                preds += predicted.tolist()

        open_file = open(file_name, "wb")
        pickle.dump(preds, open_file)
        open_file.close()

    # Compute Metrics based on the predictions
    ai.compute({"test": {"predict": preds}}, tag='regnet')

    # View the dashboard
    net.eval()
    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
    r.add_measurement()


if __name__ == '__main__':
    main()
