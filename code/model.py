import joblib

from sklearn.model_selection import GridSearchCV


class Model:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def __init__(self, model_path: str):
        self.__init__()
        self.load(model_path)

    def load(self, model_path):
        self.model = joblib.load(model_path)
        self.is_trained = True

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    def train(self, dataset):
        self.model.fit(dataset.X, dataset.y)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained! Please call either 'fit' or 'load' methods first")
        return self.model.predict(X)

    def grid_search(self, dataset, params_grid, cv=5, verbose=True):
        gs_cv = GridSearchCV(estimator=self.model, param_grid=params_grid, cv=cv, refit=True, verbose=True)
        gs_cv = gs_cv.fit(dataset.X, dataset.y)

        self.model = gs_cv.best_estimator_
        self.is_trained = True
        return gs_cv

    def __str__(self):
        return str(self.model).split('(')[0]
