import torch
import numpy as np
import time
import network
import dataset
import cost_functions
import evaluations
<<<<<<< HEAD
from early_stopping import EarlyStopping
=======
from early_stopping import EarlyStopping, LRScheduler
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
from tqdm import tqdm
import umap_functions
from torch.utils.data import SubsetRandomSampler, random_split

start_time = time.time()
<<<<<<< HEAD
seed = 2026
torch.manual_seed(seed)
np.random.seed(seed)

=======

seed = 2022
torch.manual_seed(seed)
np.random.seed(seed)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    numworkers = 0
<<<<<<< HEAD
    debugging = True  # Use smaller training partition for debugging purposes when running on personal computer
    progressbar = True
else:
    numworkers = 32
    debugging = False  # Use full partition when using the GPU
=======
    debugging = True
    progressbar = True
else:
    numworkers = 32
    debugging = False
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    progressbar = False


def main():
<<<<<<< HEAD
    umap = False  # run UMAP experiments if True
    epochs = 50
    batch_size = 64
    latent_dim = 64  # dimension of fair/private representation Z

    methods = ["IB", "CFB", "RFIB", "baseline"]
    method = 2
    '''
    Experiments done using IB, CFB, RFIB, baseline
=======
    umap = False
    epochs = 50
    batch_size = 64
    latent_dim = 32

    methods = ["IB", "CFB", "RFIB", "baseline"]
    ''' 
    Experiments done using IB, CFB, RFIB, and baseline
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    This will currently run on RFIB but can be changed with the method variable
    method 0 - IB
           1 - CFB
           2 - RFIB
           3 - baseline
    '''
<<<<<<< HEAD

    datasets = ["CelebA_skintone", "EyePACS", "fairface_race", "CelebA_gender"]
    '''
    Experiments done on CelebA, EyePACS, and FairFace datasets
    dataset_type 0 - CelebA_skintone - CelebA where sensitive variable is ITA (Skin Tone) 
                 1 - EyePACs
                 2 - FairFace
                 3 - CelebA_gender - CelebA where sensitive variable is Gender

    '''

    for dataset_type in [0,1,2]:  # Choose datasets based on above description

        # Get dataset
        if dataset_type == 0:
            train_set, test_set = dataset.get_celeba(debugging)
=======
    method = 2

    datasets = ["CelebA_race", "EyePACS", 'fairface_race']
    '''
    Experiments done on CelebA, EyePACS, and FairFace datasets
    dataset_type 0 - CelebA
                 1 - EyePACs
                 2 - FairFace
    '''
    for dataset_type in [0,1,2]:

        # Get dataset
        if dataset_type == 0:
            train_set, test_set = dataset.get_celeba()
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
        elif dataset_type == 1:
            train_set, test_set = dataset.get_eyepacs()
        elif dataset_type == 2:
            train_set, test_set = dataset.get_fairface()
<<<<<<< HEAD
        elif dataset_type == 3:
            train_set, test_set = dataset.get_celeba_gender(debugging)
=======
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        stop_early = True
        lr_schedule = False

        # Validation split for early stopping
        if stop_early or lr_schedule:

            len1, len2 = round(len(train_set) * 0.9), round(len(train_set) * 0.1)
            train_set, valset = random_split(train_set, (len1, len2))

            dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                     shuffle=True, num_workers=numworkers)

            val_dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                         shuffle=True, num_workers=numworkers)

        else:
            dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                     shuffle=True, num_workers=numworkers)

        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                      shuffle=True, num_workers=numworkers)

<<<<<<< HEAD
        # Store results
        alpha_history = []
        resultY_history = []
        result_log_history = []
        beta1_history = []
        beta2_history = []

        beta1 = 0  # only parameter for IB
        beta2s = [0.001]  # only parameter for CFB
        alphas = np.linspace(0.1, 0.9, 10)

        # Run method for various combinations of alpha and beta2
        combinations = [(a, b) for a in alphas for b in beta2s]

        for i, parameters in enumerate(combinations):
            alpha = parameters[0]
            beta2 = parameters[1]
=======
        alpha_history = []
        beta_history = []
        result_history = []
        beta1_history = []
        beta2_history = []

        beta1 = 30  # IB
        beta2 = 30  # CFB
        beta = 30
        b1orb2 = 'b1'

        # Different hyperparameter combinations
        alphas = [0.2, 0.4, 0.6, 0.8]
        betas = np.linspace(1, 50, 10)
        combinations = True
        if combinations:
            combinations = [(a, b) for a in alphas for b in betas]

        for i, parameters in enumerate(combinations):
            alpha = parameters[0]
            if b1orb2 == 'b1':
                beta1 = parameters[1]
            elif b1orb2 == 'b2':
                beta2 = parameters[1]
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

            # Get model
            if method == 3:  # Baseline
                model = network.Baseline().to(device)
<<<<<<< HEAD
            else:  # IB, CFB, RFIB
                model = network.RFIB(latent_dim).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
=======
            else:  # EyePACS, CelebA, FairFace
                model = network.RFIB(latent_dim).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            lr_scheduler = LRScheduler(optimizer)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
            early_stopping = EarlyStopping()
            loss_history = []

            # Train the network
            for epoch in range(epochs):
                train_loss = 0
                model.train()
                for x, y, s in tqdm(dataloader, disable=not progressbar):

                    model.zero_grad(set_to_none=True)

                    x = x.to(device).float()
                    y = y.to(device).float()
                    s = s.to(device).float()

                    if method == 3:  # Baseline
                        yhat = model(x)
<<<<<<< HEAD
                    else:  # IB, CFB, RFIB, or RPFIB
=======
                    else:  # EyePACs, CelebA, FairFace
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
                        yhat, yhat_fair, mu, logvar = model(x, s)

                    # IB loss
                    if method == 0:
<<<<<<< HEAD
                        loss = cost_functions.get_KLdivergence_loss(yhat, y, mu, logvar, beta1)
                    # CFB loss
                    elif method == 1:
                        loss = cost_functions.get_KLdivergence_loss(yhat_fair, y, mu, logvar, beta2)
                    # RFIB loss
                    elif method == 2:
                        loss = cost_functions.get_RFIB_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2)
=======
                        loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat, y, mu, logvar, beta)
                    # CFB loss
                    elif method == 1:
                        loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat_fair, y, mu, logvar, beta)
                    # RFIB loss
                    elif method == 2:
                        loss = cost_functions.get_combined_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
                    # Baseline loss
                    elif method == 3:
                        loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y, reduction='sum')

                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

<<<<<<< HEAD
                # Early Stopping
                if stop_early:
                    val_epoch_loss = evaluations.evaluate(model, val_dataloader, method, debugging, device, alpha,
                                                          beta1, beta2, predictions=False)
                    early_stopping(val_epoch_loss)
                    print(f'epoch: {epoch} loss: {val_epoch_loss}')

                    if early_stopping.early_stop:
                        break
=======
                # Early stopping
                if stop_early:
                    val_epoch_loss = evaluations.evaluate(model, val_dataloader, method, debugging, device, "Val", beta,
                                                          alpha, beta1, beta2)
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        break
                if lr_schedule:
                    val_epoch_loss = evaluations.evaluate(model, val_dataloader, method, debugging, device, "Val", beta,
                                                          alpha, beta1, beta2)

                    print(val_epoch_loss)
                    lr_scheduler(val_epoch_loss)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

                train_loss /= len(dataloader)
                loss_history.append(train_loss)

<<<<<<< HEAD
            if method == 3:  # Evaluate baseline
                resultY = evaluations.evaluate(model, test_dataloader, method, debugging, device, alpha, beta1, beta2,
                                               predictions=True)

            # Evaluate IB, CFB, RFIB, or RFPIB using logistic regression
            if not umap and method != 3:  # not baseline
                result_log = evaluations.evaluate_logistic_regression(model, train_set, test_set, device, debugging,
                                                                      numworkers)
=======
            # Evaluate baseline
            if method == 3 and not umap:
                result = evaluations.evaluate(model, test_dataloader, method, debugging, device, "test", beta, alpha,
                                              beta1, beta2)

            # Evaluate EyePACs, CelebA, FairFace
            elif not umap:
                result = evaluations.evaluate_logistic_regression(model, train_set, test_set, device, debugging,
                                                                  numworkers)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
            # Run UMAP experiments
            if umap and epoch >= 5:
                embedding, a_train, x_train, y_train = umap_functions.get_embedding(model, test_set, device, debugging,
                                                                                    numworkers, representation=True)
                umap_functions.plot(embedding, a_train, y_train, alpha, dataset_type, representation=True)

            # Save results
            if not umap:
                alpha_history.append(alpha)
<<<<<<< HEAD
                resultY_history.append(resultY)
                result_log_history.append(result_log)

=======
                result_history.append(result)
                beta_history.append(beta)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
                beta1_history.append(beta1)
                beta2_history.append(beta2)

                ending = ''
<<<<<<< HEAD
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_{ending}_seed_{seed}',
                        alpha_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_resultsY_{ending}_seed_{seed}',
                        resultY_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_resultslog_{ending}_seed_{seed}',
                        result_log_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_beta1s_{ending}_seed_{seed}',
                        beta1_history)

                # Save Model
                name = f'../results/{datasets[dataset_type]}_{methods[method]}_b1_{beta1}_b2_{beta2}_b3_{beta3}_{privateSensitiveEqual}_latdim_{latent_dim}.pt '
                torch.save(model.state_dict(), name)

            print("--- %s seconds ---" % (time.time() - start_time))
            print(f"dataset = {datasets[dataset_type]}")
            print(f"method = {methods[method]}")
            print(f"alpha = {alpha}")
=======
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_alphas_{ending}',
                        alpha_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_betas_{ending}', beta_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_results_{ending}',
                        result_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_beta1s_{ending}',
                        beta1_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_{b1orb2}_beta2s_{ending}',
                        beta2_history)
                torch.save(model.state_dict(),
                           f'../results/{datasets[dataset_type]}_{methods[method]}_{alpha}_{beta}_{beta1}_{beta2}_{ending}.pt')

            print("--- %s seconds ---" % (time.time() - start_time))
            print(f"beta  = {beta}")
            print(f"dataset = {datasets[dataset_type]}")
            print(f"method = {methods[method]}")
            print(f"alpha = {alpha}")
            print(f"beta = {beta}")
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
            print(f"beta1 = {beta1}")
            print(f"beta2 = {beta2}")
            print(f"numworkers = {numworkers}")
            print(f"representation_dim = {latent_dim}")
            print(f"batch size = {batch_size}")
<<<<<<< HEAD

=======
            print('\n')
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

if __name__ == '__main__':
    main()
