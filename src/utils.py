import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score, mean_squared_error


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calc_metrics(y_true, y_pred, num_features, log_target=True, return_str=True):
    n = len(y_true)
    # if Eta (cP) was transformed before
    if log_target:
        y_true, y_pred = np.exp(y_true), np.exp(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1-r2) * (n-1) / (n-num_features-1)

    result = rmse, mae, msle, mape, adj_r2
    if return_str:
        template = '-'*20 + '\nRESULT:\n\tRMSE: {:.1f}\n\tMAE:  {:.1f}\n\tMSLE: {:.4f}\n\tMAPE: {:.3f}\n\tAR2:  {:.4f}'
        return template.format(*result)

    return result


def plot_scatter(y_true, y_pred, log_target=True):
    # if Eta (cP) was transformed before
    if log_target:
        y_true, y_pred = np.exp(y_true), np.exp(y_pred)

    plt.scatter(y_true, y_pred)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True values of Eta (cP)')
    plt.ylabel('Predicted values of Eta (cP)')
