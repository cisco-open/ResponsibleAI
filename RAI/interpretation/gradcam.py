import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

import matplotlib.pyplot as plt 


class GradCAM:
    """
    Grad-CAM Visualization with respect to the class that is predicted to be most probable

    To use GradCAM with user-defined model,
    the model should be constructed as below:

    self.features_conv = nn.Sequential([
        # List of the modules until the last conv layer that will be hooked and the ReLU after that
    ])
    self.f1 = nn.Sequential([
        # List of poolings and nonlinear functions
    ])
    self.flatten = True  # True if flatten is needed for fc
    self.classifier = nn.Sequential([
        # list of FC layers for prediction
    ])
    """
    def __init__(self, model, datasets, all_classes=None, batch_size=16, device=torch.device("cpu")):
        self.model = model 
        self.net = self.model.agent
        self.net.eval()
        self.gradcamModel = GradCAMModel(self.net)

        self.all_classes = self.model.output_features[0].values if all_classes is None else all_classes
        self.n_classes = len(self.all_classes)
        self.all_classes = [self.all_classes[i] for i in range(self.n_classes)]

        self.datasets = datasets
        self.n_img = 5
        
        self.device = device

    def compute(self):
        self.setGradcamImgs()
        self.setHeatmap()
        self.visualize()

    def setGradcamImgs(self):
        dataset = self.datasets.data_dict["test"]
        dataset.get_index = True 
        batch_size = 4

        loader = DataLoader(dataset, batch_size=batch_size)

        counts = np.zeros((self.n_classes,2))  # counts[i,0] counts correct samples, counts[i,1] counts wrong samples of class i
        self.gradcamImgs = {c: {"correct": [], "wrong": [], "correct_idx": [], "wrong_idx": [], "idx": c_idx} for c_idx, c in enumerate(self.all_classes)}

        for imgs, labels, idxs in loader:
            if (counts==5).all():
                break 

            imgs = imgs.squeeze()
            
            _, predicted = torch.max(self.net(imgs), 1)
            correct = (predicted==labels).numpy()
            
            for i in range(batch_size):
                c_idx = labels[i]
                if correct[i] and counts[c_idx,0] < 5:
                    counts[c_idx,0] += 1
                    self.gradcamImgs[self.all_classes[c_idx]]["correct"].append(imgs[i])
                    self.gradcamImgs[self.all_classes[c_idx]]["correct_idx"].append(idxs[i].item())
                elif not correct[i] and counts[c_idx,1] < 5:
                    counts[c_idx,1] += 1
                    self.gradcamImgs[self.all_classes[c_idx]]["wrong"].append(imgs[i])
                    self.gradcamImgs[self.all_classes[c_idx]]["wrong_idx"].append(idxs[i].item())

        print("Grad-CAM Image Setting done")


    def setHeatmap(self):
        """
        Code based on https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
        """
        self.gradcamResults = {
            c:{
                "correct_heatmap": [],
                "correct_idx": self.gradcamImgs[c]["correct_idx"],
                "wrong_heatmap": [],
                "wrong_idx": self.gradcamImgs[c]["wrong_idx"],
                } for c in self.gradcamImgs
            }

        for c in self.gradcamImgs:
            c_dict = self.gradcamImgs[c]
            c_idx = c_dict["idx"]

            for img in c_dict["correct"]:
                img = img.unsqueeze(0)
                pred = self.gradcamModel(img)
                pred[0, c_idx].backward()
                gradients = self.gradcamModel.get_activations_gradient()
                pooled_gradients = torch.mean(gradients, dim=[2,3])
                activations = self.gradcamModel.get_activations(img).detach()
                activations = activations * pooled_gradients.unsqueeze(-1).unsqueeze(-1)
                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
                heatmap /= torch.max(heatmap)
                self.gradcamResults[c]["correct_heatmap"].append(heatmap.numpy())

            for img in c_dict["wrong"]:
                img = img.unsqueeze(0)
                pred = self.gradcamModel(img)
                pred[0, c_idx].backward()
                gradients = self.gradcamModel.get_activations_gradient()
                pooled_gradients = torch.mean(gradients, dim=[2,3])
                activations = self.gradcamModel.get_activations(img).detach()
                activations = activations * pooled_gradients.unsqueeze(-1).unsqueeze(-1)
                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
                heatmap /= torch.max(heatmap)
                self.gradcamResults[c]["wrong_heatmap"].append(heatmap.numpy())
        
        print("Grad-CAM Compute Done")


    def visualize(self):
        self.viz_results = {c:{"correct": [], "wrong": []} for c in self.gradcamImgs}
        dataset = self.datasets.data_dict["test"]
        
        for c in self.viz_results:
            c_dict = self.gradcamResults[c]
            for heatmap, idx in zip(c_dict["correct_heatmap"], c_dict["correct_idx"]):
                img = dataset.getRawItem(idx).squeeze()

                img_size = (img.shape[1], img.shape[2])
                resized_heatmap = cv2.resize(heatmap, img_size)
                resized_heatmap = np.uint8(resized_heatmap*255)
                resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                img = np.transpose(np.uint8(img*255), (1,2,0))  #CHW to HWC

                superimposed_img = (0.3*resized_heatmap + 0.7*img).astype(np.uint8) 
                self.viz_results[c]["correct"].append((img, superimposed_img))

        for c in self.viz_results:
            c_dict = self.gradcamResults[c]
            for heatmap, idx in zip(c_dict["wrong_heatmap"], c_dict["wrong_idx"]):
                img = dataset.getRawItem(idx).squeeze()

                img_size = (img.shape[2], img.shape[1])
                resized_heatmap = cv2.resize(heatmap, img_size)
                resized_heatmap = np.uint8(resized_heatmap*255)
                resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                img = np.transpose(np.uint8(img*255), (1,2,0))  #CHW to HWC

                superimposed_img = (0.3*resized_heatmap + 0.7*img).astype(np.uint8) 
                self.viz_results[c]["wrong"].append((img, superimposed_img))

        print("Grad-CAM Visualize Done")

        ### FOR SANITY CHECK 

        plt.imshow(self.viz_results["dog"]["correct"][0][0])
        plt.savefig("./dog_correct_img_0.png")
        plt.close()
        plt.imshow(self.viz_results["dog"]["correct"][0][1])
        plt.savefig("./dog_correct_heatmap_0.png")
        plt.close()
        




class GradCAMModel(nn.Module):
    def __init__(self, net):
        super(GradCAMModel, self).__init__()

        self.net = net 
        self.hook()

    def hook(self):
        if "torchvision" in self.net.__class__.__module__:
            self.hookTorchvisionModel()
        else:
            self.hookUserDefinedModel()

    def hookTorchvisionModel(self):
        if self.net.__class__.__name__ in ["VGG"]:
            self.features_conv = self.net.features[:36]  #vgg19

    def hookUserDefinedModel(self):
        self.features_conv = self.net.features_conv 
        self.f1 = self.net.f1 
        self.flatten = self.net.flatten
        self.classifier = self.net.classifier 

    def activations_hook(self, grad):
        self.gradients = grad 
    
    def forward(self, x):
        x = self.features_conv(x)
        x.register_hook(self.activations_hook)

        x = self.f1(x)
        if self.flatten:
            x = torch.flatten(x,1)
        x = self.classifier(x)
        return x 

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)

