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
from data import FootballDataset


def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


filenames = []
for root, dir, files in os.walk('data/football'):
    for file in files:
        # print(file)
        if not file.startswith('.'):
            filenames.append(file)

football_dataset = FootballDataset('data/football', filenames)

fig = plt.figure()

for i in range(len(football_dataset)):
    sample, obejct_type = football_dataset[i]
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_image(sample)

    if i == 3:
        plt.show()
        break


data_transforms = transforms.Compose([
        transforms.Resize((224, 224))])

fig = plt.figure()
sample, object_type = football_dataset[0]
transformed_sample = data_transforms(sample)
ax = plt.subplot(1, 3, 1)
plt.tight_layout()
show_image(transformed_sample)

plt.show()


transformed_dataset = FootballDataset('data/football', filenames,
                                      transform=transforms.Compose([transforms.Resize((224,224)),
                                                                     ToTensor()
                                                                     ]))

for i in range(len(transformed_dataset)):
    sample, obejct_type = transformed_dataset[i]
    print(i, sample.size(), obejct_type)
    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=2, shuffle=True, num_workers=0)
#
def show_images_batch(sample_batched):
    images_batch, type_batch = sample_batched
    bathc_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))

    plt.title('Batch from dataloader')


# torch.Size([64, 3, 299, 299]) torch.Size([64])

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, len(sample_batched))
    print(sample_batched[0].shape, sample_batched[1].shape)
    if i_batch == 3:
        plt.figure()
        show_images_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break