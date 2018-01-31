from __future__ import print_function
from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):

    def __init__(self, x_train, y_train, train=True):
        self.train = train
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        if self.train:
            return self.x_train[index], self.y_train[index] # img = Image.fromarray(img)
        else:
            pass


    def __len__(self):
        if self.train:
            return len(self.x_train)
        else:
            pass
