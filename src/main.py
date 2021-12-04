import torch
import numpy as np
import time
import network
import dataset
import cost_functions
import evaluations
from early_stopping import EarlyStopping, LRScheduler
from tqdm import tqdm
import umap_functions
from torch.utils.data import SubsetRandomSampler, random_split

start_time = time.time()

seed = 2022
torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    numworkers = 0
    debugging = True
    progressbar = True
else:
    numworkers = 32
    debugging = False
    progressbar = False


def main():
    umap = False
    epochs = 50
    batch_size = 64
    latent_dim = 32

    methods = ["IB", "CFB", "RFIB", "baseline"]
    ''' 
    Experiments done using IB, CFB, RFIB, and baseline
    This will currently run on RFIB but can be changed with the method variable
    method 0 - IB
           1 - CFB
           2 - RFIB
           3 - baseline
    '''
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
        elif dataset_type == 1:
            train_set, test_set = dataset.get_eyepacs()
        elif dataset_type == 2:
            train_set, test_set = dataset.get_fairface()

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

            # Get model
            if method == 3:  # Baseline
                model = network.Baseline().to(device)
            else:  # EyePACS, CelebA, FairFace
                model = network.RFIB(latent_dim).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            lr_scheduler = LRScheduler(optimizer)
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
                    else:  # EyePACs, CelebA, FairFace
                        yhat, yhat_fair, mu, logvar = model(x, s)

                    # IB loss
                    if method == 0:
                        loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat, y, mu, logvar, beta)
                    # CFB loss
                    elif method == 1:
                        loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat_fair, y, mu, logvar, beta)
                    # RFIB loss
                    elif method == 2:
                        loss = cost_functions.get_combined_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2)
                    # Baseline loss
                    elif method == 3:
                        loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y, reduction='sum')

                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

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

                train_loss /= len(dataloader)
                loss_history.append(train_loss)

            # Evaluate baseline
            if method == 3 and not umap:
                result = evaluations.evaluate(model, test_dataloader, method, debugging, device, "test", beta, alpha,
                                              beta1, beta2)

            # Evaluate EyePACs, CelebA, FairFace
            elif not umap:
                result = evaluations.evaluate_logistic_regression(model, train_set, test_set, device, debugging,
                                                                  numworkers)
            # Run UMAP experiments
            if umap and epoch >= 5:
                embedding, a_train, x_train, y_train = umap_functions.get_embedding(model, test_set, device, debugging,
                                                                                    numworkers, representation=True)
                umap_functions.plot(embedding, a_train, y_train, alpha, dataset_type, representation=True)

            # Save results
            if not umap:
                alpha_history.append(alpha)
                result_history.append(result)
                beta_history.append(beta)
                beta1_history.append(beta1)
                beta2_history.append(beta2)

                ending = ''
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
            print(f"beta1 = {beta1}")
            print(f"beta2 = {beta2}")
            print(f"numworkers = {numworkers}")
            print(f"representation_dim = {latent_dim}")
            print(f"batch size = {batch_size}")
            print('\n')

if __name__ == '__main__':
    main()
