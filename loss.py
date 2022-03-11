import numpy as np
def loss_01(y_true, y_pred):
    return np.sum(y_pred != y_true) / len(y_pred)

def equal_oppo_loss(y_true, y_pred):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return np.sum(y_pred_arr[y_true_arr == 1] != 1) / len(y_pred_arr[y_true_arr == 1])
