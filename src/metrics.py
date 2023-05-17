import numpy as np
import torch

def get_accuracy(predictions,y):
    accuracy = np.sum(predictions == y) / len(y)

    return accuracy


def get_min_accuracy(predictions, y, s):

    predictions_0 = predictions[s == 0]
    y_0 = y[s == 0]
    accuracy_0 = np.sum(predictions_0 == y_0) / (len(y_0) + 1e-10)

    predictions_1 = predictions[s == 1]
    y_1 = y[s == 1]
    accuracy_1 = np.sum(predictions_1 == y_1) / (len(y_1) + 1e-10)

    return accuracy_0, accuracy_1


def get_py_given_s(predictions,S, s):
    return np.sum((predictions == 1) & (S == s)) / (np.sum(S == s) + 1e-10)

def get_discrimination(predictions, S):


    categorical = np.any(S > 1) # true if categorical sensitive attribute
    # print(f'categorical {categorical}')

    if not categorical:
        pos_predictions_a1 = np.sum((predictions == 1) & (S == 1))
        pos_predictions_a0 = np.sum((predictions == 1) & (S == 0))

        a1 = np.sum(S == 1)
        a0 = np.sum(S == 0)

        discrimination = np.abs(pos_predictions_a1 / (a1 + 1e-10) - pos_predictions_a0 / (a0 + 1e-10))
    else: #fairface categorical
        s0 = get_py_given_s(predictions,S, 0)
        s1 = get_py_given_s(predictions,S, 1)
        s2 = get_py_given_s(predictions,S, 2)

        # print(f's0 = {s0}')
        # print(f's1 = {s1}')
        # print(f's2 = {s2}')

        d1 = np.abs(s0-s1)
        d2 = np.abs(s0-s2)
        d3 = np.abs(s1-s0)
        d4 = np.abs(s1-s2)
        d5 = np.abs(s2-s0)
        d6 = np.abs(s2-s1)

        discrimination = (d1+d2+d3+d4+d5+d6) / 6


    return discrimination

def get_di_ratio(predictions,s):
    # print(f'predictions {predictions}')
    pos_predictions_a1 = np.sum((predictions == 1) & (s == 1))
    pos_predictions_a0 = np.sum((predictions == 1) & (s == 0))

    a1 = np.sum(s == 1)
    a0 = np.sum(s == 0)

    Pr_y_hat_1_z_0 = pos_predictions_a0 / (a0 + 1e-10)
    Pr_y_hat_1_z_1 = pos_predictions_a1 / (a1 + 1e-10)
    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)

    return min_dp / max_dp

def get_equalized_odds_gap(predictions, y, s):


    y0 = (y == 0)
    y1 = (y == 1)
    # print(f's = {s}')
    # print(f'y1 = {y1}')
    #
    # print(f's[y1] = {s[y1]}')

    disc1 = get_discrimination(predictions[y1], s[y1])
    disc0 = get_discrimination(predictions[y0], s[y0])

    return max(disc0,disc1)




def get_eo_ratio(predictions,y,s):
    y1 = (y == 1)
    pos_predictions_a0 = np.sum((predictions[y1] == 1) & (s[y1] == 0))
    pos_predictions_a1 = np.sum((predictions[y1] == 1) & (s[y1] == 1))
    
    a0 = np.sum(s[y1] == 0)
    a1 = np.sum(s[y1] == 1)


    Pr_y_hat_1_z_0 = pos_predictions_a0 / (a0 + 1e-10)
    Pr_y_hat_1_z_1 = pos_predictions_a1 / (a1 + 1e-10)
    min_eo = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    max_eo = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)


    return min_eo/max_eo


def get_acc_gap(predictions, y, s):

    categorical = np.any(s > 1) # true if categorical sensitive attribute
    # print(f'categorical {categorical}')

    a1 = (s == 1)
    a0 = (s == 0)

    acc1 = np.sum(predictions[a1] == y[a1]) / (len(y[a1]) + 1e-10)
    acc0 = np.sum(predictions[a0] == y[a0]) / (len(y[a0]) + 1e-10)

    if not categorical:
        return np.abs(acc1-acc0)
    else:
        a2 = (s == 2)

        acc2 = np.sum(predictions[a2] == y[a2]) / (len(y[a2]) + 1e-10)

        d1 = np.abs(acc0-acc1)
        d2 = np.abs(acc0-acc2)
        d3 = np.abs(acc1-acc0)
        d4 = np.abs(acc1-acc2)
        d5 = np.abs(acc2-acc0)
        d6 = np.abs(acc2-acc1)

        return (d1+d2+d3+d4+d5+d6) / 6
