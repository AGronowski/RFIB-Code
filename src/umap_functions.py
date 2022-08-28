import umap
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# UMAP plots
<<<<<<< HEAD
def plot(embedding, s_train, y_train, alpha, dataset_type, representation=False):
=======
def plot(embedding, a_train, y_train, alpha, dataset_type, representation=False):
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

    datasets = ["CelebA_race", "EyePACS", 'fairface_race']
    dataset = datasets[dataset_type]

    method_name = 'RFIB'

    save = True
    show = False

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
<<<<<<< HEAD
        c=[sns.color_palette()[int(x)] for x in s_train])
=======
        c=[sns.color_palette()[int(x)] for x in a_train])
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    plt.gca().set_aspect('equal', 'datalim')
    if representation:
        plt.title(rf'{dataset} RFIB $\alpha = ${alpha} S', fontsize=24)
    else:
        plt.title(rf'{dataset} original data A', fontsize=24)

    if save:
        if representation:
            plt.savefig(f"../results/umapplots/umap_{dataset}_{method_name}_{alpha}_A.png")
        else:
            plt.savefig(f"../results/umapplots/umap_{dataset}_original_data_A.png")
    if show:
        plt.show()

    plt.clf()

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[int(x)+8] for x in y_train])
    plt.gca().set_aspect('equal', 'datalim')
    if representation:
        plt.title(rf'{dataset} {method_name} $\alpha = ${alpha} Y', fontsize=24)
    else:
        plt.title(rf'{dataset} original data Y', fontsize=24)
    if save:
        if representation:
            plt.savefig(f"../results/umapplots/umap_{dataset}_{method_name}_{alpha}_Y.png")
        else:
            plt.savefig(f"../results/umapplots/umap_{dataset}_original_data_Y.png")
    if show:
        plt.show()
    plt.clf()


# Run UMAP
def get_embedding(model,dataset,device,debugging,numworkers,representation=False):
    model.eval()
    reducer = umap.UMAP()

    with torch.no_grad():
        # Train
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                     shuffle=True, num_workers=numworkers)
        x_list = []
        y_list = []
        z_list = []
<<<<<<< HEAD
        s_list = []

        for x, y, s in tqdm(dataloader, disable=not (debugging)):
            x = x.to(device).float()
            y = y.to(device).float()
            s = s.to(device).float()
=======
        a_list = []

        for x, y, a in tqdm(dataloader, disable=not (debugging)):

            x = x.to(device).float()
            y = y.to(device).float()
            a = a.to(device).float()
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

            z, mu, logvar = model.getz(x)

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
<<<<<<< HEAD
            s_list.append(s)
=======
            a_list.append(a)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        X_train = torch.cat(x_list,dim=0)
        Z_train = torch.cat(z_list,dim=0)
        Y_train = torch.cat(y_list,dim=0)
<<<<<<< HEAD
        S_train = torch.cat(s_list,dim=0)
=======
        A_train = torch.cat(a_list,dim=0)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        X_train = X_train.cpu()
        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()
<<<<<<< HEAD
        S_train = S_train.cpu()
=======
        A_train = A_train.cpu()
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)

        if representation:
            embedding = reducer.fit_transform(Z_train)
            X_train = Z_train
        else:
            scaled_X_train = StandardScaler().fit_transform(X_train)
            embedding = reducer.fit_transform(scaled_X_train)

<<<<<<< HEAD
        return embedding, S_train, X_train, Y_train
=======
        return embedding, A_train, X_train, Y_train
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
