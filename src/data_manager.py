import numpy as np
import pandas as pd


class DataManager:
    def __init__(self, data_path, log_target=True, add_fe=False):
        # placeholders
        self.X, self.y = None, None

        self.X_cols, self.y_col = ['T2LM (ms)', 'T (K)', 'TE (ms)'], 'Eta (cP)'
        self.log_target = log_target

        self.load(data_path)

        if add_fe:
            self.add_feature_engineering()

    def load(self, data_path):
        # load CSV files
        df = pd.read_csv(data_path, index_col=0)
        self.X, self.y = df[self.X_cols], df[self.y_col]

        # transform target Eta (cP) -> log(Eta)
        if self.log_target:
            new_y_col = 'log(Eta)'
            self.y = np.log(self.y)
            self.y.name = self.y_col = new_y_col

    def add_feature_engineering(self):
        if self.X is None:
            raise ValueError("Dataset is not loaded. Please call 'load' method first")

        self.X['log(T2LM)'] = np.log(self.X['T2LM (ms)'])
        self.X['log(T)/TE'] = np.log(self.X['T (K)']) / self.X['TE (ms)']
        self.X['log(T2LM)/TE'] = np.log(self.X['T2LM (ms)']) / self.X['TE (ms)']
        self.X['log(T)'] = np.log(self.X['T (K)'])

        self.X_cols = self.X_cols[:-1] + ['log(T2LM)', 'log(T)/TE', 'log(T2LM)/TE', 'log(T)']
        self.X = self.X[self.X_cols]
