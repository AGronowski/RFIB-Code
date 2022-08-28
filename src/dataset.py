import torch
import torchvision
from torchvision import transforms
from skimage import io
import numpy as np
import pandas as pd
import os

torch.manual_seed(2022)
np.random.seed(2022)

<<<<<<< HEAD
class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, gender=False):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.gender = gender
=======

class CelebaDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

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
<<<<<<< HEAD

        if self.gender:  # S = Gender
            sensitive = self.frame.iloc[index, 3]  # Sensitive is gender
        else:  # S = ITA (Skin Tone)
            sensitive = self.frame.iloc[index, 4]  # Sensitive is skin tone
=======
        sensitive = self.frame.iloc[index, 4]  # Sensitive is race
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

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

<<<<<<< HEAD
        target = self.frame.iloc[index, 2]  # Target is Diabetic Retinopathy Status
        sensitive = self.frame.iloc[index, 4]  # Sensitive is ITA (Skin Tone)
=======
        target = self.frame.iloc[index, 2]  # Target is diabetic_retinopathy
        sensitive = self.frame.iloc[index, 4]  # Sensitive is ita_dark
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

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
<<<<<<< HEAD
                                self.frame.iloc[index, 3])
=======
                                self.frame.iloc[index, 1])
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
        image = io.imread(img_name)
        image = torchvision.transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

<<<<<<< HEAD
        target = self.frame.iloc[index, 6]  # Y = Gender
        sensitive = self.frame.iloc[index, 5]  # S = ITA (Skin Tone)
=======
        target = self.frame.iloc[index, 7]  # Y = male
        sensitive = self.frame.iloc[index, 9]  # S = race_black
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        return image, target, sensitive


def get_fairface():
    root_dir = "../data/fairface/"

    transform = transforms.Compose([transforms.ToTensor(),
<<<<<<< HEAD
                                    # first - mean for RGG channels, 2nd std for RGB channels
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])

    # Predict gender, race is sensitive
    trainset = FairfaceDataset('../data/fairface_train_good_oct27.csv', root_dir, transform)
    testset = FairfaceDataset('../data/fairface_test_good_oct27.csv', root_dir, transform)
=======
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    # Predict gender, race is sensitive
    trainset = FairfaceDataset('../data/fairface_train.csv', root_dir, transform)
    testset = FairfaceDataset('../data/fairface_val.csv', root_dir, transform)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

    return trainset, testset


def get_eyepacs():
    root_dir = "../data/eyepacs"

    image_size = 256
<<<<<<< HEAD

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),  # converts 0-255 to 0-1
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])

    trainset = EyepacsDataset('../data/eyepacs_control_train_jpeg.csv', root_dir,
=======
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    csv_file = '../data/eyepacs_control_train_jpeg.csv'
    trainset = EyepacsDataset(csv_file, root_dir,
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
                              transform)
    testset = EyepacsDataset('../data/eyepacs_test_dr_ita_jpeg.csv', root_dir,
                             transform)

    return trainset, testset

<<<<<<< HEAD
# CelebA dataset where S = ITA (skin tone)
def get_celeba(debugging):
=======

def get_celeba():
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    root_dir = '../data/celeba'
    image_size = 128
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
<<<<<<< HEAD
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])
    if debugging:
        trainset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform, gender=False)
        testset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform, gender=False)
    else:
        #  Y = Age, S = Race
        trainset = CelebaDataset('../data/celeba_skincolor_train_jpg.csv', root_dir, transform, gender=False)
        testset = CelebaDataset('../data/celeba_balanced_combo_test_jpg.csv', root_dir, transform, gender=False)

    return trainset, testset

# CelebA dataset where S = gender
def get_celeba_gender(debugging):
    root_dir = '../data/celeba'
    image_size = 128

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])
    if debugging:
        trainset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform, gender=True)
        testset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform, gender=True)
    else:
        #  Y = Age, S = Gender
        trainset = CelebaDataset('../data/celeba_gender_train_jpg.csv', root_dir, transform, gender=True)
        testset = CelebaDataset('../data/celeba_balanced_combo_test_jpg.csv', root_dir, transform, gender=True)

    return trainset, testset
=======
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    #  Y = Age, S = Race
    trainset = CelebaDataset('../data/celeba_skincolor_train_jpg.csv', root_dir, transform)
    testset = CelebaDataset('../data/celeba_balanced_combo_test_jpg.csv', root_dir, transform)

    return trainset, testset



>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
