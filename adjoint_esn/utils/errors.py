import numpy as np


def L2(y, y_pred, axis=None):
    # equivalent to np.sqrt(np.sum((y-y_pred)**2, axis = axis))
    # frobenius norm is default
    return np.linalg.norm(y - y_pred, axis=axis)


def rel_L2(y, y_pred, axis=None):
    return L2(y, y_pred, axis=axis) / np.linalg.norm(y, axis=axis)


def mse(y, y_pred, axis=None):
    # mean squared error
    return np.mean((y - y_pred) ** 2, axis=axis)


def rmse(y, y_pred, axis=None):
    # root mean squared error
    return np.sqrt(mse(y, y_pred, axis))


def nrmse(y, y_pred, axis=None, normalize_by="rms"):
    # normalized root mean squared error
    if normalize_by == "rms":
        norm = np.sqrt(np.mean(y) ** 2, axis=axis)
    elif normalize_by == "maxmin":
        norm = np.max(y, axis=axis) - np.min(y, axis=axis)

    return rmse(y, y_pred, axis) / norm
