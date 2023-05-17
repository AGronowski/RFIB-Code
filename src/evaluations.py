import torch
from tqdm import tqdm
import numpy as np
import metrics
import cost_functions
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy
from sklearn import preprocessing
import pandas as pd


def evaluate_logistic_regression(model, trainset, testset, device, debugging, numworkers):
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    with torch.no_grad():

        # Train
        traindataloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)
        y_list = []
        z_list = []

        for x, y, s, p in tqdm(traindataloader, disable=not (debugging)):
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
        s_list = []

        for x, y, s, p in tqdm(testdataloader, disable=not (debugging)):
            x = x.to(device).float()
            y = y.to(device).float()
            s = s.to(device).float()

            z, mu, logvar = model.getz(x)

            y_list.append(y)
            z_list.append(z)
            s_list.append(s)

        Z_test = torch.cat(z_list, dim=0)
        Y_test = torch.cat(y_list, dim=0)
        S_test = torch.cat(s_list, dim=0)

        Z_test = Z_test.cpu()
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        predictions = np.argmax(predictions, 1)
        y = Y_test.cpu().detach().numpy()
        s = S_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions, y)
        accgap = metrics.get_acc_gap(predictions, y, s)
        dpgap = metrics.get_discrimination(predictions, s)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, s)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, s)
        di_ratio = metrics.get_di_ratio(predictions,s)
        eo_ratio = metrics.get_eo_ratio(predictions,y,s)


        print(f"logistic accuracy = {accuracy}")
        print(f"logistic accgap = {accgap}")
        print(f"logistic dpgap = {dpgap}")
        print(f"logistic di_ratio = {di_ratio}")
        print(f"logistic eo_ratio = {eo_ratio}")
        print(f"logistic eqoddsgap = {eqoddsgap}")
        print(f"logistic acc_min_0 = {accmin0}")
        print(f"logistic acc_min_1 = {accmin1}")

        return np.array([accuracy, accgap, dpgap, eqoddsgap, accmin0, accmin1,di_ratio,eo_ratio])


# Get Loss
def evaluate(model, dataloader, method, debugging, device, alpha, beta1, beta2,beta3,predictions=True):

    model.eval()
    testloss = 0

    y_list = []
    s_list = []
    p_list = []
    yhat_list = []
    yhat_fair_list = []

    with torch.no_grad():
        for x, y, s, p in tqdm(dataloader, disable=not debugging):
            x = x.to(device).float()
            y = y.to(device).float()
            s = s.to(device).float()
            p = p.to(device).float()

            if method == 3:  # baseline
                yhat = model(x)
            else:
                yhat, yhat_fair, mu, logvar, reconstruction = model(x, s, p)
                yhat_fair_list.append(yhat_fair)

            y_list.append(y)
            s_list.append(s)
            p_list.append(p)
            yhat_list.append(yhat)

        # get loss for validation
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
            # baseline loss
            elif method == 3:
                loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y,
                                                                reduction='sum')
            testloss += loss.item()

        if predictions:
            Y_test = torch.cat(y_list, dim=0)
            S_test = torch.cat(s_list, dim=0)
            Yhat_test = torch.cat(yhat_list, dim=0)

            y = Y_test.cpu().detach().numpy()
            s = S_test.cpu().detach().numpy()

            # yhat predictions
            predictions = Yhat_test.cpu().view(-1).detach().numpy()
            predictions[predictions < 0.5] = 0
            predictions[predictions > 0.5] = 1


            accuracy = metrics.get_accuracy(predictions, y)
            accgap = metrics.get_acc_gap(predictions, y, s)
            dpgap = metrics.get_discrimination(predictions, s)
            eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, s)
            accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, s)
            di_ratio = metrics.get_di_ratio(predictions, s)
            eo_ratio = metrics.get_eo_ratio(predictions, y, s)
            print('yhat predicting y:\n')
            print(f" accuracy = {accuracy}")
            print(f" accgap = {accgap}")
            print(f" dpgap = {dpgap}")
            print(f" eqoddsgap = {eqoddsgap}")
            print(f" acc_min_0 = {accmin0}")
            print(f" acc_min_1 = {accmin1}")

            return np.array([accuracy, accgap, dpgap, eqoddsgap, accmin0, accmin1,di_ratio,eo_ratio])



        else:  # validation loss when predictions = False
            print('val')
            return testloss


#only for testing eyepacs aa. regular train, modified test
def evaluate_logistic_regression_eyepacs_aa(model, trainset, testset, device, debugging, numworkers):
    # sets model in evalutation mode
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    # predictor = sklearn.ensemble.RandomForestClassifier()
    with torch.no_grad():
        '''train '''
        # X_train,Y_train,A_train = trainset.images, trainset.targets, trainset.sensitives
        # X_train = X_train.to(device)
        # Y_train = Y_train.to(device)

        traindataloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)

        y_list = []
        z_list = []

        for x, y, s in tqdm(traindataloader, disable=not (debugging)):
            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1

            z, mu, logvar = model.getz(x)

            y_list.append(y)
            z_list.append(z)

        Z_train = torch.cat(z_list, dim=0)
        Y_train = torch.cat(y_list, dim=0)

        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()

        scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_scaled = scaler.transform(Z_train)
        # predictor.fit(Z_train.flatten(start_dim=1), Y_train)

        # try to prevent nan or inf error
        # torch.where: (condition, condition true, condition false)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        predictor.fit(Z_scaled, Y_train)

        ''' test '''

        testdataloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                     shuffle=False, num_workers=numworkers)

        y_list = []
        z_list = []
        image_list = []

        for x, y, image in tqdm(testdataloader, disable=not (debugging)):
           x = x.to(device).float()  # batch size x input_dim
           y = y.to(device).float()  # batch size x 1

           z, mu, logvar = model.getz(x)

           y_list.append(y)
           z_list.append(z)
           image_list.append(list(image))

        Z_test = torch.cat(z_list,dim=0)
        Y_test = torch.cat(y_list,dim=0)
        images = np.concatenate(image_list)

        Z_test = Z_test.cpu()
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        predictions = np.argmax(predictions,1)
        y = Y_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions,y)

        combined_lists = [images,y,predictions]
        frame = pd.DataFrame(combined_lists)


        print(f"logistic accuracy = {accuracy}")


        return frame.transpose()



