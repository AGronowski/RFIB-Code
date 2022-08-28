import torch
from tqdm import tqdm
import numpy as np
import metrics
import cost_functions
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy
from sklearn import preprocessing


<<<<<<< HEAD
# Run logistic regression classifier on Z
=======
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
def evaluate_logistic_regression(model, trainset, testset, device, debugging, numworkers):
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    with torch.no_grad():

        # Train
        traindataloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)
        y_list = []
        z_list = []

<<<<<<< HEAD
        for x, y, s in tqdm(traindataloader, disable=not (debugging)):
=======
        for x, y, a in tqdm(traindataloader, disable=not (debugging)):
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
            x = x.to(device).float()
            y = y.to(device).float()

            z, mu, logvar = model.getz(x)

            y_list.append(y)
            z_list.append(z)

        Z_train = torch.cat(z_list, dim=0)
        Y_train = torch.cat(y_list, dim=0)

        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()

        scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_scaled = scaler.transform(Z_train)

        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        predictor.fit(Z_scaled, Y_train)

        # Test
        testdataloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=numworkers)

        y_list = []
        z_list = []
<<<<<<< HEAD
        s_list = []

        for x, y, s in tqdm(testdataloader, disable=not (debugging)):
            x = x.to(device).float()
            y = y.to(device).float()
            s = s.to(device).float()
=======
        a_list = []

        for x, y, a in tqdm(testdataloader, disable=not (debugging)):
            x = x.to(device).float()
            y = y.to(device).float()
            a = a.to(device).float()
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

            z, mu, logvar = model.getz(x)

            y_list.append(y)
            z_list.append(z)
<<<<<<< HEAD
            s_list.append(s)

        Z_test = torch.cat(z_list, dim=0)
        Y_test = torch.cat(y_list, dim=0)
        S_test = torch.cat(s_list, dim=0)
=======
            a_list.append(a)

        Z_test = torch.cat(z_list, dim=0)
        Y_test = torch.cat(y_list, dim=0)
        A_test = torch.cat(a_list, dim=0)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        Z_test = Z_test.cpu()
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        predictions = np.argmax(predictions, 1)
        y = Y_test.cpu().detach().numpy()
<<<<<<< HEAD
        s = S_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions, y)
        accgap = metrics.get_acc_gap(predictions, y, s)
        dpgap = metrics.get_discrimination(predictions, s)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, s)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, s)
=======
        a = A_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions, y)
        accgap = metrics.get_acc_gap(predictions, y, a)
        dpgap = metrics.get_discrimination(predictions, a)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, a)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, a)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        print(f"logistic accuracy = {accuracy}")
        print(f"logistic accgap = {accgap}")
        print(f"logistic dpgap = {dpgap}")
        print(f"logistic eqoddsgap = {eqoddsgap}")
        print(f"logistic acc_min_0 = {accmin0}")
        print(f"logistic acc_min_1 = {accmin1}")

        return np.array([accuracy, accgap, dpgap, eqoddsgap, accmin0, accmin1])


# Get Loss
<<<<<<< HEAD
def evaluate(model, dataloader, method, debugging, device, alpha, beta1, beta2, predictions=True):
=======
def evaluate(model, dataloader, method, debugging, device, description, beta=1, alpha=1, beta1=1, beta2=1):

>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
    model.eval()
    testloss = 0

    y_list = []
<<<<<<< HEAD
    s_list = []
    yhat_list = []
    yhat_fair_list = []

    with torch.no_grad():
        for x, y, s in tqdm(dataloader, disable=not debugging):
            x = x.to(device).float()
            y = y.to(device).float()
            s = s.to(device).float()
=======
    a_list = []
    yhat_list = []

    with torch.no_grad():
        for x, y, a in tqdm(dataloader, disable=not debugging):
            x = x.to(device).float()
            y = y.to(device).float()
            a = a.to(device).float()
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

            if method == 3:  # baseline
                yhat = model(x)
            else:
<<<<<<< HEAD
                yhat, yhat_fair, mu, logvar = model(x, s)
                yhat_fair_list.append(yhat_fair)

            y_list.append(y)
            s_list.append(s)
            yhat_list.append(yhat)

        # Get loss for validation
        if not predictions:

            # IB loss
            if method == 0:
                loss = cost_functions.get_KLdivergence_loss(yhat, y, mu, logvar, beta1)
            # CFB loss
            elif method == 1:
                loss = cost_functions.get_KLdivergence_loss(yhat_fair, y, mu, logvar, beta2)
            # RFIB loss
            elif method == 2:
                loss = cost_functions.get_RFIB_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2)
=======
                yhat, yhat_fair, mu, logvar = model(x, a)

            y_list.append(y)
            a_list.append(a)
            yhat_list.append(yhat)

            # IB loss
            if method == 0:
                loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat, y, mu, logvar, beta)
            # CFB loss
            elif method == 1:
                loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat_fair, y, mu, logvar, beta)
            # RFIB loss
            elif method == 2:
                loss = cost_functions.get_combined_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1, beta2)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
            # baseline loss
            elif method == 3:
                loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y,
                                                                reduction='sum')

            testloss += loss.item()

<<<<<<< HEAD
        if predictions:
            Y_test = torch.cat(y_list, dim=0)
            A_test = torch.cat(s_list, dim=0)
            Yhat_test = torch.cat(yhat_list, dim=0)

            y = Y_test.cpu().detach().numpy()
            s = A_test.cpu().detach().numpy()

            # yhat predictions
            predictions = Yhat_test.cpu().view(-1).detach().numpy()
            predictions[predictions < 0.5] = 0
            predictions[predictions > 0.5] = 1

            accuracy = metrics.get_accuracy(predictions, y)
            accgap = metrics.get_acc_gap(predictions, y, s)
            dpgap = metrics.get_discrimination(predictions, s)
            eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, s)
            accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, s)

            print('yhat predicting y:\n')
=======
        Y_test = torch.cat(y_list, dim=0)
        A_test = torch.cat(a_list, dim=0)
        Yhat_test = torch.cat(yhat_list, dim=0)

        y = Y_test.cpu().detach().numpy()
        a = A_test.cpu().detach().numpy()
        predictions = Yhat_test.cpu().view(-1).detach().numpy()
        predictions[predictions < 0.5] = 0
        predictions[predictions > 0.5] = 1

        accuracy = metrics.get_accuracy(predictions, y)
        accgap = metrics.get_acc_gap(predictions, y, a)
        dpgap = metrics.get_discrimination(predictions, a)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, a)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, a)

        if description != 'Val':
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
            print(f" accuracy = {accuracy}")
            print(f" accgap = {accgap}")
            print(f" dpgap = {dpgap}")
            print(f" eqoddsgap = {eqoddsgap}")
            print(f" acc_min_0 = {accmin0}")
            print(f" acc_min_1 = {accmin1}")
<<<<<<< HEAD

            return np.array(
                [round(accuracy, 6), round(accgap, 6), round(dpgap, 6), round(eqoddsgap, 6), round(accmin0, 6),
                 round(accmin1, 6)])

        else:  # Validation loss when predictions = False
            return testloss

=======
        print(f"{description}\n")

        # baseline and not used for validation
        if method == 3 and description != 'Val':
            return np.array(
                [round(accuracy, 6), round(accgap, 6), round(dpgap, 6), round(eqoddsgap, 6), round(accmin0, 6),
                 round(accmin1, 6)])
        else:
            return testloss
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
