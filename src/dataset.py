import torch
import torchvision
from torchvision import transforms
from skimage import io
import numpy as np
import pandas as pd
import os
import main
from sklearn import preprocessing


torch.manual_seed(2022)
np.random.seed(2022)

private_sensitive_equal = main.privateSensitiveEqual
swapVariables = main.swapVariables

if private_sensitive_equal:
    print('private sensitive equal')
else:
    print('private sensitive NOT equal')


class CelebaDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None,gender=False):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.gender = gender


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

        if self.gender:
            sensitive = self.frame.iloc[index, 3]  # Sensitive is gender
            private = self.frame.iloc[index, 4]
        else:
            sensitive = self.frame.iloc[index, 4]  # Sensitive is race
            private = self.frame.iloc[index, 3]

        if private_sensitive_equal:
            private = sensitive

        if swapVariables:
            temp = sensitive
            sensitive = private
            private = temp

        return image, target, sensitive, private


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

        return image, target, sensitive, sensitive

#for testing with the aa clinician labels testset
class Eyepacs_race_test_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # self.targets = self.frame.iloc[:,2]
        # self.sensitives = self.frame.iloc[:, 4]
        # self.images = images
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name) #numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image) #PIL image

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index,2] #target is diabetic_retinopathy

        return image, target, self.frame.iloc[index, 1] #name


class FairfaceDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        # img_name = os.path.join(self.root_dir,
        #                         self.frame.iloc[index, 1])  #fairface_val or fairface_train

        # img_name = os.path.join(self.root_dir,
        #                         self.frame.iloc[index, 3]) #oct27 csvs
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1]) #may6 csvs
        image = io.imread(img_name)
        image = torchvision.transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)


        # target = self.frame.iloc[index, 7]  # Y = male   #fairface_val or fairface_train
        # sensitive = self.frame.iloc[index, 9]  # S = race_black

        # target = self.frame.iloc[index, 6]  # Y = male   #oct27 csvs
        # sensitive = self.frame.iloc[index, 5]  # S = race_black

        sensitive = self.frame.iloc[index, 4] #may6 csvs
        target = self.frame.iloc[index, 3]  # gender
        # print(img_name)
        # print(target)
        if target == "Male":
            target = 1.0
        else:
            target = 0.0
        # print(target)


        if sensitive == "Black":
            sensitive = 0.0
        elif sensitive == "Latino_Hispanic":
            sensitive = 1.0
        else:
            sensitive = 2.0



        return image, target, sensitive, sensitive


class Adult_dataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, data_hidden, transform=None, task='fairness'):
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
        self.target_vals = 2
        self.hidden_vals = 2
        self.task = task

    def __getitem__(self, index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        if self.task == 'fairness':
            target, hidden = int(target), int(hidden)
        else:
            hidden = int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden, hidden

    def __len__(self):
        return len(self.targets)
def get_fairface():
    root_dir = "../data/fairface/"

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 # first - mean for RGG channels, 2nd std for RGB channels
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])
    transform = transforms.Compose([transforms.ToTensor(),
                                    # first - mean for RGG channels, 2nd std for RGB channels
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])


    # Predict gender, race is sensitive
    # trainset = FairfaceDataset('../data/fairface_train.csv', root_dir, transform)
    # testset = FairfaceDataset('../data/fairface_test.csv', root_dir, transform)
    trainset = FairfaceDataset('../data/fairface_train_cat.csv', root_dir, transform)
    testset = FairfaceDataset('../data/fairface_test_cat.csv', root_dir, transform)
    return trainset, testset


def get_eyepacs():
    root_dir = "../data/eyepacs"

    image_size = 256
    # transform = transforms.Compose([transforms.Resize(image_size),
    #                                 transforms.CenterCrop(image_size),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(), #converts 0-255 to 0-1
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)), #does nothing (subtract 0, divide by 1)
                                    ])

    trainset = EyepacsDataset('../data/eyepacs_control_train_jpeg.csv', root_dir,
                              transform)
    testset = EyepacsDataset('../data/eyepacs_test_dr_ita_jpeg.csv', root_dir,
                             transform)

    return trainset, testset


def get_celeba(debugging):
    root_dir = '../data/celeba'
    image_size = 128
    # transform = transforms.Compose([transforms.Resize(image_size),
    #                                 transforms.CenterCrop(image_size),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])
    if debugging:
        trainset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform,gender=False)
        testset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform,gender=False)
    else:
        #  Y = Age, S = Race
        trainset = CelebaDataset('../data/celeba_skincolor_train_jpg.csv', root_dir, transform,gender=False)
        testset = CelebaDataset('../data/celeba_balanced_combo_test_jpg.csv', root_dir, transform,gender=False)

    return trainset, testset

def get_celeba_gender(debugging):
    root_dir = '../data/celeba'
    image_size = 128
    # transform = transforms.Compose([transforms.Resize(image_size),
    #                                 transforms.CenterCrop(image_size),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                    ])
    if debugging:
        trainset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform,gender=True)
        testset = CelebaDataset('../data/celeba_debugging.csv', root_dir, transform,gender=True)
    else:
        #  Y = Age, S = Gender
        trainset = CelebaDataset('../data/celeba_gender_train_jpg.csv', root_dir, transform,gender=True)
        testset = CelebaDataset('../data/celeba_balanced_combo_test_jpg.csv', root_dir, transform,gender=True)

    return trainset, testset

# returns ordinary test dataset, aa testset with special dataloader
def get_testaa_eyepacs():
    root_dir = "../data/eyepacs_aa"

    image_size = 256

    transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]) #scales to [-1,1]

    #REMOVE THE DEBUGGING
    # csv_file = '../data/eyepacs_debugging.csv'
    csv_file = '../data/eyepacs_control_train_jpeg.csv'

    root_dir = "../data/eyepacs"
    trainset = EyepacsDataset(csv_file,root_dir,
                              transform)
    # testset =  Eyepacs_race_test_dataset('../data/eyepacs_debugging.csv',root_dir,
    #                               transform)


    root_dir = "../data/eyepacs_aa"
    testset = Eyepacs_race_test_dataset('../data/test_dr_aa_jpeg.csv',root_dir,
                                  transform)


    return trainset, testset


def get_adult(task='fairness'):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'salary']
    dummy_variables = {
        'workclass': [
            'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'],
        'education': ['Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, \
            12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'],
        'education-num': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'],
        'marital-status': ['Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,\
            Married-AF-spouse'],
        'occupation': ['Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, \
            Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, \
            Protective-serv, Armed-Forces'],
        'relationship': ['Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'],
        'race': ['White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'],
        'sex': ['Female, Male'],
        'native-country': ['United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), \
            India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, \
            Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, \
            Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands']
    }
    for k in dummy_variables:
        dummy_variables[k] = [v.strip() for v in dummy_variables[k][0].split(',')]

    # Load Adult dataset

    # https: // archive.ics.uci.edu / ml / datasets / Adult
    data_train = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=column_names,header=None
    )
    data_test = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        names=column_names,skiprows=1,header=None
    )

    # data_train = pd.read_csv(
    #     '../data/adult_us_cens/adult.data',
    #     names=column_names, header=None
    # )
    # data_test = pd.read_csv(
    #     '../data/adult_us_cens/adult.test',
    #     names=column_names, skiprows=1, header=None
    # )
    data_train = data_train.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    data_test = data_test.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)

    def get_variables(data, task='fairness'):

        le = preprocessing.LabelEncoder()
        dummy_columns = list(dummy_variables.keys())
        dummy_columns.remove('sex')
        data[dummy_columns] = data[dummy_columns].apply(lambda col: le.fit_transform(col))
        X = data.drop('sex', axis=1).drop('salary', axis=1).to_numpy().astype(float)
        S = data['sex'].to_numpy()
        if task == 'fairness':
            T = data['salary'].to_numpy()
            T = np.where(np.logical_or(T == '<=50K', T == '<=50K.'), 0, 1)
        else:
            T = data.drop('sex', axis=1).drop('salary', axis=1).to_numpy().astype(float)
        S = np.where(S == 'Male', 0, 1)

        return X, S, T

    X_train, S_train, T_train = get_variables(data_train, task)
    X_test, S_test, T_test = get_variables(data_test, task)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    X_train = (X_train - X_mean) / (X_std)
    X_test = (X_test - X_mean) / (X_std)
    if task == 'privacy':
        for i in range(len(T_train[1, :])):
            if len(np.unique(T_train[:, i])) > 42:
                t_mean, t_std = T_train[:, i].mean(), T_train[:, i].std()
                T_train[:, i] = (T_train[:, i] - t_mean) / t_std
                T_test[:, i] = (T_test[:, i] - t_mean) / t_std

    trainset = Adult_dataset(X_train, T_train, S_train, task=task)
    testset = Adult_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset


class Compas_dataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, data_hidden, transform=None, task='fairness'):
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
        self.target_vals = 2
        self.hidden_vals = 2
        self.task = task

    def __getitem__(self, index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        if self.task == 'fairness':
            target, hidden = int(target), int(hidden)
        else:
            hidden = int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden, hidden

    def __len__(self):
        return len(self.targets)


def get_compas(task='fairness'):
    data = pd.read_csv(
        '../data/propublica_data_for_fairml.csv',
        header=0, sep=','
    )

    msk = np.zeros(len(data))
    msk[:int(0.7 * len(data))] = 1
    msk = np.random.permutation(msk).astype('bool')
    data_train = data[msk]
    data_test = data[~msk]

    def get_variables(_data):
        X = _data.drop('Two_yr_Recidivism', axis=1).drop('African_American', axis=1).to_numpy()
        S = _data['African_American'].to_numpy()
        T = _data['Two_yr_Recidivism'].to_numpy()

        return X, S, T

    X_train, S_train, T_train = get_variables(data_train)
    X_test, S_test, T_test = get_variables(data_test)

    if task == 'privacy':
        T_train = X_train
        T_test = X_test

    mean, std = X_train[:, 0].mean(), X_train[:, 0].std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    trainset = Compas_dataset(X_train, T_train, S_train, task=task)
    testset = Compas_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset