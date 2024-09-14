from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestModel:
    def __init__(self, params=None):
        self.model = RandomForestClassifier()
        self.params = params if params else {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search = None

    def train(self, X_train, y_train, cv=5):
        self.grid_search = GridSearchCV(self.model, self.params, cv=cv, n_jobs=-1)
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test):
        return self.grid_search.predict(X_test)

    def get_best_params(self):
        return self.grid_search.best_params_

    def feature_importance(self):
        return self.grid_search.best_estimator_.feature_importances_
