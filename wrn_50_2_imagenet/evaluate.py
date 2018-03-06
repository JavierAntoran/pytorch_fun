from __future__ import division

import os
import time
import copy
from sys import argv

import torch
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from src.utils import *
from define_model import my_wrn_transfer
from sys import argv


#################
# Import Data and transformations
#################

batch_size = 32

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dogscats'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes



def run_test(model):
    since = time.time()

    # Each epoch has a training and validation phase
    phase = 'val'

    model.train(False)  # Set model to evaluate mode

    running_corrects = 0

        # Iterate over data.
    for data in dataloaders[phase]:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_cuda:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects / dataset_sizes[phase]

    print('{} Acc: {:.4f}'.format(
        phase, epoch_acc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

#

use_cuda = False #torch.cuda.is_available()  # torch.cuda.is_available()
#

model_name = argv[1]

params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')
net = my_wrn_transfer(params, 2, use_cuda)
#net.cuda()
net.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
print('it works!!')

run_test(net)
#probs = F.softmax(vals, dim=1)


