import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
import collections

# For data normalization
# left tuple = means for the 3 channels of the dataset
# right tuple = std deviation of the 3 channels
# 0.5s are approximations for cifar10 mean and
# std_dev values over 3 channels (r, g, b)
# exact values can be found here
# https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data

data_stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# size of dataset batch used for training
# change to change the amount of images used max is 10000
# note that the higher the number
# the less visible the images will be during preview
batch_size = 64
resize_size = (16, 16)

test_transform = transforms.Compose(
    [
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        # normalization follows a [-1, 1] range
        # with (data = (data * mean)/std_dev)
        transforms.Normalize(*data_stats, inplace=True),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # normalization follows a [-1, 1] range
        # with (data = (data * mean)/std_dev)
        transforms.Normalize(*data_stats, inplace=True),
    ]
)

# note that cifar10 is essentially a numpy array
# pin_memory speeds up the loading onto GPU from CPU

cifar_trainset = datasets.CIFAR10(
    root="./data/", train=True, download=True, transform=train_transform
)

train_loader = DataLoader(
    cifar_trainset, batch_size=batch_size, shuffle=True, pin_memory=True
)

cifar_testset = datasets.CIFAR10(
    root="./data/", train=False, download=True, transform=test_transform
)

test_loader = DataLoader(
    cifar_testset, batch_size=batch_size, shuffle=True, pin_memory=True
)


# array of all possible categories from cifar10
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# code to show image after normalization
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# uses matplotlib to display a preview of the images
def preview_images(data):
    dataiter = iter(data)
    images, labels = next(dataiter)
    imshow(make_grid(images))
    labelDict = collections.defaultdict(int)
    for lab in labels:
        labelDict[classes[lab]] += 1
    print("Images in preview:")
    for entry in labelDict:
        print(f"{entry}: {labelDict[entry]}")


preview_images(test_loader)
