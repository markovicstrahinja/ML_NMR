import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score, mean_squared_error


def calc_metrics(y_true, y_pred, log_target=True):
    # if Eta (cP) was transformed before
    if log_target:
        y_true, y_pred = np.exp(y_true), np.exp(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    msle = mean_squared_log_error(y_true, y_pred)
    # + adj_r2, MAPE

    return mae, rmse, msle, 0., 0.


def plot_scatter(y_true, y_pred, log_target=True):
    # if Eta (cP) was transformed before
    if log_target:
        y_true, y_pred = np.exp(y_true), np.exp(y_pred)

    plt.scatter(y_true, y_pred)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True values of Eta (cP)')
    plt.ylabel('Predicted values of Eta (cP)')