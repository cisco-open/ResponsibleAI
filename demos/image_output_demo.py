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
# This demo uses Cifar10 dataset and shows how RAI can be used to evaluate image classification tasks


# importing modules
import os
import sys
import inspect
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim

from dotenv import load_dotenv
from torchvision import datasets, transforms
# https://colab.research.google.com/drive/1Ozin9zX89xfoyn63o5B7l5bR5W_E1oy0#scrollTo=7hAFf5Ue4VP_

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.db.service import RaiDB
from RAI.dataset import Dataset, Feature, MetaDatabase, NumpyData
from RAI.utils import torch_to_RAI

#setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

load_dotenv(f'{currentdir}/../.env')

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)


class Object(object):
    pass


config = Object()
config.batch_size = 128
config.epochs = 1
config.lr = 0.0002
config.beta1 = 0.5
config.nz = 100
config.ngf = 64
config.ndf = 64
config.ngpu = 1
config.nc = 3
config.image_size = 32
config.workers = 2
config.no_cuda = False
config.seed = manualSeed
config.log_interval = 10


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.nz, config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.ngf * 8, config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.ngf * 4, config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.ngf * 2, config.nc, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, input):
        return self.main(input)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def train(gen, disc, device, dataloader, optimizerG, optimizerD, criterion, epoch, iters):
  gen.train()
  disc.train()
  img_list = []
  fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)

  # Establish convention for real and fake labels during training (with label smoothing)
  real_label = 0.9
  fake_label = 0.1
  for i, data in enumerate(dataloader, 0):
      disc.zero_grad()
      real_cpu = data[0].to(device)
      b_size = real_cpu.size(0)
      label = torch.full((b_size,), real_label, device=device)
      output = disc(real_cpu).view(-1)
      errD_real = criterion(output, label)
      errD_real.backward()
      D_x = output.mean().item()

      ## Train with all-fake batch
      noise = torch.randn(b_size, config.nz, 1, 1, device=device)
      fake = gen(noise)
      label.fill_(fake_label)
      output = disc(fake.detach()).view(-1)
      errD_fake = criterion(output, label)
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      errD = errD_real + errD_fake
      optimizerD.step()

      # Update Generator
      gen.zero_grad()
      label.fill_(real_label)
      output = disc(fake).view(-1)
      errG = criterion(output, label)
      errG.backward()
      D_G_z2 = output.mean().item()

      optimizerG.step()

      if i % 5 == 0:
          print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, config.epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
      iters += 1


def produce_gan():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=config.workers)
    netG = Generator(config.ngpu).to(device)
    if (device.type == 'cuda') and (config.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(config.ngpu)))
    netG.apply(weights_init)
    netD = Discriminator(config.ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (config.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(config.ngpu)))
    netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    iters = 0
    for epoch in range(1, config.epochs + 1):
        train(netG, netD, device, trainloader, optimizerG, optimizerD, criterion, epoch, iters)
    torch.save(netG.state_dict(), "cifar_gan.h5")


def main():
    use_dashboard = True

    gan = Generator(config.ngpu)
    PATH = "./cifar_gan.h5"
    if not os.path.isfile(PATH):
        print("Training GAN")
        produce_gan()

    print("Loading model")
    gan.load_state_dict(torch.load(PATH))

    def generate_fake_image():
        noise = torch.randn(1, config.nz, 1, 1)
        return [gan(noise).detach().numpy()]

    generated = None
    for i in range(50):
        img = generate_fake_image()
        if generated is None:
            generated = img
        else:
            generated = np.vstack((generated, img))

    output = Feature("Cifar Image", "image", "CIFAR Image produced by GAN")
    model = Model(agent=gan, output_features=output, name="gan", generate_image_fun=generate_fake_image,
                  description="Text Summarizer", model_class="gan")
    configuration = {"time_complexity": "polynomial"}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    xTestData, yTestData, raw = torch_to_RAI(testloader)

    #setup the dataset
    dataset = Dataset({"cifar": NumpyData(None, xTestData, raw)})
    meta = MetaDatabase([])
    
    # initialize RAI 
    ai = AISystem(name="cifar_gan_x_y", task='generate', meta_database=meta, dataset=dataset, model=model)
    ai.initialize(user_config=configuration)
    ai.compute({"cifar": {"generate_image": generated}}, tag='epoch_1_generations')

    if use_dashboard:
        r = RaiDB(ai)
        r.reset_data()
        r.export_metadata()
        r.add_measurement()
        r.export_visualizations("cifar", "cifar")

    ai.display_metric_values()

    from RAI.Analysis import AnalysisManager
    analysis = AnalysisManager()
    print("available analysis: ", analysis.get_available_analysis(ai, "test"))
    '''
    result = analysis.run_all(ai, "test", "Test run!")
    for analysis in result:
        print("Analysis: " + analysis)
        print(result[analysis].to_string())
    '''


if __name__ == '__main__':
    main()
