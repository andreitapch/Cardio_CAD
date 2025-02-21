from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from itertools import product
from sklearn.neural_network import MLPClassifier

class ModelParams:
    def get_param_grid(self, class_type, unbalanced_type):
        if class_type == 'KNN_class':
            self.param_grid = {
                'n_neighbors': list(range(2, 31)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'braycurtis','cosine']  #
            }
            self.clf = KNeighborsClassifier()
        elif class_type == 'DT_class':
            self.param_grid = {
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5],
                'ccp_alpha': [0.0, 0.01]
            }
            self.clf = DecisionTreeClassifier(random_state=42)

        elif class_type == 'RF_class':
            if unbalanced_type == 'None':

                self.param_grid = {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [5, 7, 10, 15, None],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [2, 4, 8],
                    'class_weight': ['balanced', 'balanced_subsample']
                    }
            else:
                self.param_grid = {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [5, 7, 10, 15, None],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [2, 4, 8],
                }

            self.clf = RandomForestClassifier(random_state=42)

        elif class_type == 'GBM_class':
            # self.param_grid = {
            #     'n_estimators': [100, 200, 300, 500],
            #     'learning_rate': [0.01, 0.05, 0.1],
            #     'max_depth': [3, 4, 5, 6],
            #     'subsample': [0.8, 1.0],
            #     'min_samples_split': [5, 10, 20]
            # }
            self.param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'min_samples_split': [10, 20]
            }

            self.clf = GradientBoostingClassifier(random_state=42)

        elif class_type == 'XGBoost_class':
            self.param_grid = {
                'n_estimators': [100],
                'learning_rate': [0.05],
                'max_depth': [5],
                # 'subsample': [0.8],
                # 'colsample_bytree': [1.0],
                # 'min_child_weight': [1],
                # 'scale_pos_weight': [25]
            }

            # self.param_grid = {
            #     'n_estimators': [100, 200, 300, 500],
            #     'learning_rate': [0.01, 0.05, 0.1],
            #     'max_depth': [3, 4, 5, 6],
            #     'subsample': [0.8, 1.0],
            #     'colsample_bytree': [0.8, 1.0],
            #     'min_child_weight': [1, 3, 5],
            #     'scale_pos_weight': [1, 10, 25, 50]
            # }
            self.clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')


        elif class_type == 'LightGBM_class':
            self.param_grid = {
                'n_estimators': [100, 200, 300],  # Keep fewer values to reduce time
                'learning_rate': [0.01, 0.05, 0.1],
                # 'max_depth': [4, 6],  # Allow unlimited depth (-1)
                'num_leaves': [20, 40, 60],  # Reduced upper bound
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.9, 1.0],
                'min_child_samples': [5, 10, 20],  # Reduced values
                # 'scale_pos_weight': [5, 10, 25],  # Instead of `class_weight='balanced'`
            }

            self.clf = LGBMClassifier(random_state=42)

        # elif class_type == 'CatBoost_class':
        #     self.param_grid = {
        #         'iterations': [200, 300],
        #         'learning_rate': [0.05, 0.1],
        #         'depth': [4, 5],
        #         'l2_leaf_reg': [3, 5],
        #         'scale_pos_weight': [10, 25, 50]
        #     }
        #     self.clf = CatBoostClassifier(random_state=42, verbose=0)

        elif class_type == 'SVM_class':
            self.param_grid = {
                'C': [1, 10, 100],
                'kernel': ['poly', 'rbf', 'sigmoid'],
                'gamma': ['scale'],
                'class_weight': [None, 'balanced']
            }

            self.clf = SVC(probability=True, random_state=42)
        elif class_type == 'MLP_class':
            self.param_grid = {
                # 'hidden_layer_sizes': [(50,), (100,), (200,)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'max_iter': [10000],
            }
            self.clf = MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=100)
        # elif class_type == 'LogisticRegression_class':
        #     self.param_grid = {
        #         'C': [0.1, 1, 10],
        #         'penalty': ['l2'],
        #         'solver': ['lbfgs', 'saga'],
        #         'max_iter': [200],
        #         'class_weight': ['balanced']
        #     }
        #     self.clf = LogisticRegression(random_state=42)

        else:
            raise ValueError(f"Unknown classifier type: {class_type}")
        self.param_combinations = [dict(zip(self.param_grid.keys(), v)) for v in product(*self.param_grid.values())]