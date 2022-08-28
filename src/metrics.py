import numpy as np


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


def get_discrimination(predictions, s):
    pos_predictions_a1 = np.sum((predictions == 1) & (s == 1))
    pos_predictions_a0 = np.sum((predictions == 1) & (s == 0))

    a1 = np.sum(s == 1)
    a0 = np.sum(s == 0)

    discrimination = np.abs(pos_predictions_a1 / (a1 + 1e-10) - pos_predictions_a0 / (a0 + 1e-10))
    return discrimination


def get_equalized_odds_gap(predictions, y, s):
    y0 = (y == 0)
    y1 = (y == 1)

    disc1 = get_discrimination(predictions[y1], s[y1])
    disc0 = get_discrimination(predictions[y0], s[y0])

    return max(disc0,disc1)


def get_acc_gap(predictions, y, s):
    a1 = (s == 1)
    a0 = (s == 0)

    acc1 = np.sum(predictions[a1] == y[a1]) / (len(y[a1]) + 1e-10)
    acc0 = np.sum(predictions[a0] == y[a0]) / (len(y[a0]) + 1e-10)

    return np.abs(acc1-acc0)