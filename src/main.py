import torch
import numpy as np
import network
import dataset
import cost_functions
import evaluations
from early_stopping import EarlyStopping
from tqdm import tqdm
import umap_functions
from torch.utils.data import SubsetRandomSampler, random_split

seed = 2026
torch.manual_seed(seed)
np.random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    numworkers = 0
    debugging = True  # Use smaller training partition for debugging purposes when running on personal computer
    progressbar = True
else:
    numworkers = 32
    debugging = False  # Use full partition when using the GPU
    progressbar = False


def main():
    umap = False  # run UMAP experiments if True
    epochs = 50
    batch_size = 64
    latent_dim = 64  # dimension of fair/private representation Z

    methods = ["IB", "CFB", "RFIB", "baseline"]
    method = 2
    '''
    Experiments done using IB, CFB, RFIB, baseline
    This will currently run on RFIB but can be changed with the method variable
    method 0 - IB
           1 - CFB
           2 - RFIB
           3 - baseline
    '''

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
        elif dataset_type == 1:
            train_set, test_set = dataset.get_eyepacs()
        elif dataset_type == 2:
            train_set, test_set = dataset.get_fairface()
        elif dataset_type == 3:
            train_set, test_set = dataset.get_celeba_gender(debugging)

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

            # Get model
            if method == 3:  # Baseline
                model = network.Baseline().to(device)
            else:  # IB, CFB, RFIB
                model = network.RFIB(latent_dim).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
                    else:  # IB, CFB, RFIB
                        yhat, yhat_fair, mu, logvar = model(x, s)

                    # IB loss
                    if method == 0:
                        loss = cost_functions.get_KLdivergence_loss(yhat, y, mu, logvar, beta1)
                    # CFB loss
                    elif method == 1:
                        loss = cost_functions.get_KLdivergence_loss(yhat_fair, y, mu, logvar, beta2)
                    # RFIB loss
                    elif method == 2:
                        loss = cost_functions.get_RFIB_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2)
                    # Baseline loss
                    elif method == 3:
                        loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y, reduction='sum')

                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

                # Early Stopping
                if stop_early:
                    val_epoch_loss = evaluations.evaluate(model, val_dataloader, method, debugging, device, alpha,
                                                          beta1, beta2, predictions=False)
                    early_stopping(val_epoch_loss)
                    print(f'epoch: {epoch} loss: {val_epoch_loss}')

                    if early_stopping.early_stop:
                        break

                train_loss /= len(dataloader)
                loss_history.append(train_loss)

            if method == 3:  # Evaluate baseline
                resultY = evaluations.evaluate(model, test_dataloader, method, debugging, device, alpha, beta1, beta2,
                                               predictions=True)

            # Evaluate IB, CFB, or RFIB using logistic regression
            if not umap and method != 3:  # not baseline
                result_log = evaluations.evaluate_logistic_regression(model, train_set, test_set, device, debugging,
                                                                      numworkers)
            # Run UMAP experiments
            if umap and epoch >= 5:
                embedding, a_train, x_train, y_train = umap_functions.get_embedding(model, test_set, device, debugging,
                                                                                    numworkers, representation=True)
                umap_functions.plot(embedding, a_train, y_train, alpha, dataset_type, representation=True)

            # Save results
            if not umap:
                alpha_history.append(alpha)
                resultY_history.append(resultY)
                result_log_history.append(result_log)

                beta1_history.append(beta1)
                beta2_history.append(beta2)

                ending = ''
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas_{ending}_seed_{seed}',
                        alpha_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_resultsY_{ending}_seed_{seed}',
                        resultY_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_resultslog_{ending}_seed_{seed}',
                        result_log_history)
                np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_beta1s_{ending}_seed_{seed}',
                        beta1_history)

                # Save Model
                name = f'../results/{datasets[dataset_type]}_{methods[method]}_b1_{beta1}_b2_{beta2}_{privateSensitiveEqual}_latdim_{latent_dim}.pt '
                torch.save(model.state_dict(), name)

            print(f"dataset = {datasets[dataset_type]}")
            print(f"method = {methods[method]}")
            print(f"alpha = {alpha}")
            print(f"beta1 = {beta1}")
            print(f"beta2 = {beta2}")
            print(f"numworkers = {numworkers}")
            print(f"representation_dim = {latent_dim}")
            print(f"batch size = {batch_size}")


if __name__ == '__main__':
    main()
