# usage: train.py [-h] [-d DATA_PATH] [-m MODEL_NAME] [-j PARAMS_JSON] [-g] [-l]
#                 [-f] [-v]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -d DATA_PATH, --data-path DATA_PATH
#                         Name of input CSV data file.
#   -m MODEL_NAME, --model-name MODEL_NAME
#                         File name for weights saving. If None then
#                         "models/{model_class}_weights.joblib" model name will be used.
#   -j PARAMS_JSON, --params-json PARAMS_JSON
#                         Path to JSON with model and training params. base-
#                         model: path to joblib base model;grid-search: params
#                         for grid search (only needed if -grid-search=True)
#   -g, --grid-search     if yes then make a grid search over params in
#                         --params-json file (for example, see field "grid-
#                         search" in svm_params.json file)
#   -l, --log-target      If yes then build prediction for log(Eta) instead if Eta.
#   -f, --feature-engineering
#                         If yes then use feature engineering
#   -v, --verbose         Print verbose output.

import argparse
import joblib
import json


from src.data_manager import DataManager
from src.model import Model
from src.utils import calc_metrics
import test


def parse_args():
    parser = argparse.ArgumentParser()

    # Location params
    parser.add_argument('-d', '--train-path', dest='train_path', default='data/train_data.csv',
                        help='Name of input CSV data file for training.')
    parser.add_argument('-t', '--test-path', dest='test_path', default='data/test_data.csv',
                        help='Name of input CSV data file for training.')
    parser.add_argument('-m', '--model-name', dest='model_name', type=str,
                        help='File name for weights saving. '
                             'If None then "models/{model_class}_weights.joblib" model name will be used.')
    parser.add_argument('-j', '--params-json', dest='params_json', default='svr_params.json',
                        help='Path to JSON with model and training params. '
                             'base-model: path to joblib base model;'
                             'grid-search: params for grid search (only needed if -grid-search=True)')

    # Training params
    parser.add_argument('-g', '--grid-search', dest='grid_search', action='store_true', default=False,
                        help='if yes then make a grid search over params in --params-json file '
                             '(for example, see field "grid-search" in svm_params.json file)')
    parser.add_argument('-l', '--log-target', dest='log_target', action='store_true', default=False,
                        help='If yes then build prediction for log(Eta) instead if Eta.')
    parser.add_argument('-f', '--feature-engineering', dest='feature_engineering', action='store_true', default=False,
                        help='If yes then use feature engineering')

    # Other params
    parser.add_argument('-n', '--n-jobs', dest='n_jobs', default=-1, type=int,
                        help='Number of parallel jobs in grid-search (if flag grid-search is active)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output.')

    return parser.parse_args()


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
def train(model: Model, data: DataManager,  test_data: DataManager,
          output_model_name=None, grid_search_params=None, n_jobs=-1, verbose=True):
    if grid_search_params:
        if verbose: print('Start grid search..', end='')
        for c in [1e-2, 1e-1, 1, 2, 5, 10]:
            for coef0 in [0, 0.1, 0.5, 1.0]:
                # for degree in [1, 2, 3, 4]:
                    # for gamma in ["scale", 5e-4, 1e-3, 1e-1]:
                        # for epsilon in [1e-4, 1e-3, 1e-2, 1e-0]:
                for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
                    model = SVR(
                        C=c, cache_size=200, coef0=coef0, degree=3, epsilon=0.001, gamma='scale',
                        kernel=kernel, max_iter=-1, shrinking=True, tol=0.001, verbose=False
                    )
                    model.fit(data.X, data.y)
                    y_pred = model.predict(test_data.X)
                    res = calc_metrics(test_data.y, y_pred, len(data.X_cols), log_target=data.log_target)
                    res_num = calc_metrics(test_data.y, y_pred, len(data.X_cols), log_target=data.log_target, return_str=False)
                    if res_num[-2] < 50 and res_num[0] < 10000:
                        print(c, coef0, kernel, res)

        # model.grid_search(data, grid_search_params, n_jobs, verbose=verbose)
        if verbose: print('Grid search finished.')
    else:
        if verbose: print('Start training..', end='')
        model.train(dataset=data)
        if verbose: print('Training finished.')

    print('TRAIN ', end='')
    test.test(model, data, verbose=verbose)

    print('TEST ', end='')
    test.test(model, test_data, verbose=verbose)

    if output_model_name is None:
        output_model_name = f'models/{model}_weights.joblib'
    model.save(output_model_name)
    if verbose: print('Estimator:\n', model.model)
    if verbose: print(f'Model saved in {output_model_name}')


if __name__ == '__main__':
    args = parse_args()

    # Data loading
    if args.verbose: print('Data loading..', end='')
    train_data_manager = DataManager(args.train_path, args.log_target, args.feature_engineering)
    test_data_manager = DataManager(args.test_path, args.log_target, args.feature_engineering)
    if args.verbose: print('Done')

    # Model loading
    if args.verbose: print('Model loading..', end='')
    with open(args.params_json, 'r') as f:
        params_json = json.load(f)

    base_model = Model(params_json['base-model'])
    if args.verbose: print('Done')

    gs_params = None
    if args.grid_search:
        gs_params = params_json['grid-search']

    train(base_model, train_data_manager, test_data_manager, args.model_name, gs_params, args.n_jobs, args.verbose)
