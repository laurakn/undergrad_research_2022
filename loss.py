import numpy as np
def loss_01(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_pred)

def equal_oppo_loss(y_pred, y_true):
    p = y_true == 1
    return np.sum(y_pred[p] != 1) / len(y_pred[p])
