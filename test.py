# usage: test.py [-h] [-d DATA_PATH] [-m MODEL_NAME] [-l] [-f] [-v]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -d DATA_PATH, --data-path DATA_PATH
#                         Name of input CSV data file.
#   -m MODEL_NAME, --model-name MODEL_NAME
#                         Trained file name for loading.
#   -l, --log-target      If yes then make prediction for log(Eta) instead of Eta.
#   -f, --feature-engineering
#                         If yes then use feature engineering
#   -v, --verbose         Print verbose output.

import argparse
import joblib
import json

from src.data_manager import DataManager
from src.model import Model
from src.utils import calc_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    # Location params
    parser.add_argument('-d', '--data-path', dest='data_path', default='data/test_data.csv',
                        help='Name of input CSV data file.')
    parser.add_argument('-m', '--model-name', dest='model_name', type=str, help='Trained file name for loading.')

    # Data params
    parser.add_argument('-l', '--log-target', dest='log_target', action='store_true', default=False,
                        help='If yes then make prediction for log(Eta) instead if Eta.')
    parser.add_argument('-f', '--feature-engineering', dest='feature_engineering', action='store_true', default=False,
                        help='If yes then use feature engineering')

    # Other params
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output.')

    return parser.parse_args()


def test(model: Model, data: DataManager, verbose=True):
    if verbose: print('Start predicting..', end='')
    y_pred = model.predict(data.X)
    if verbose: print('Done')

    result = calc_metrics(data.y, y_pred, len(data.X_cols), data.log_target, return_str=True)
    print(result)


if __name__ == '__main__':
    args = parse_args()

    # Data loading
    if args.verbose: print('Data loading..', end='')
    data_manager = DataManager(args.data_path, args.log_target, args.feature_engineering)
    if args.verbose: print('Done')

    # Model loading
    if args.verbose: print('Model loading..', end='')
    model = Model(args.model_name)
    if args.verbose: print('Done')

    test(model, data_manager, args.verbose)
