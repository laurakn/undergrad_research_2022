import numpy as np
def loss_01(y_pred, y_true):
    return np.sum(y_pred != y_true) / y_pred.shape[0]