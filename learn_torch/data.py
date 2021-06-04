from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor, Lambda, Compose
from PIL import Image


OBJECT_MAP = {'player': 0, 'referee': 1, 'ball': 2}


class FootballDataset(Dataset):
    """"football objects"""

    def __init__(self, dir, filenames, transform=None):
        """
        :param dir:
        """
        self.dir = dir
        self.transform = transform
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.dir, self.filenames[idx])
        image = Image.open(img_name)
        object_name = self.filenames[idx].split('_')[0]
        mapped_object = OBJECT_MAP[object_name]

        if self.transform:
            image = self.transform(image)

        return image, mapped_object
