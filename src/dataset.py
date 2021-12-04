import torch
import torchvision
from torchvision import transforms
from skimage import io
import numpy as np
import pandas as pd
import os

torch.manual_seed(2022)
np.random.seed(2022)


class CelebaDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name)
        image = torchvision.transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index, 2]  # Target is age
        sensitive = self.frame.iloc[index, 4]  # Sensitive is race

        return image, target, sensitive


class EyepacsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name)
        image = torchvision.transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index, 2]  # Target is diabetic_retinopathy
        sensitive = self.frame.iloc[index, 4]  # Sensitive is ita_dark

        return image, target, sensitive


class FairfaceDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name)
        image = torchvision.transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index, 7]  # Y = male
        sensitive = self.frame.iloc[index, 9]  # S = race_black

        return image, target, sensitive


def get_fairface():
    root_dir = "../data/fairface/"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    # Predict gender, race is sensitive
    trainset = FairfaceDataset('../data/fairface_train.csv', root_dir, transform)
    testset = FairfaceDataset('../data/fairface_val.csv', root_dir, transform)

    return trainset, testset


def get_eyepacs():
    root_dir = "../data/eyepacs"

    image_size = 256
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    csv_file = '../data/eyepacs_control_train_jpeg.csv'
    trainset = EyepacsDataset(csv_file, root_dir,
                              transform)
    testset = EyepacsDataset('../data/eyepacs_test_dr_ita_jpeg.csv', root_dir,
                             transform)

    return trainset, testset


def get_celeba():
    root_dir = '../data/celeba'
    image_size = 128
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    #  Y = Age, S = Race
    trainset = CelebaDataset('../data/celeba_skincolor_train_jpg.csv', root_dir, transform)
    testset = CelebaDataset('../data/celeba_balanced_combo_test_jpg.csv', root_dir, transform)

    return trainset, testset



