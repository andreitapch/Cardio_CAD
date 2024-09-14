import xgboost as xgb
from sklearn.model_selection import GridSearchCV


class GradientBoostingModel:
    def __init__(self, params=None):
        self.model = xgb.XGBClassifier()
        self.params = params if params else {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
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
