from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from itertools import product


class ModelParams:
    def __init__(self):
        self.clf = None
        self.param_grid = None
        self.param_combinations = None

    def get_param_grid(self, class_type, unbalanced_type):
        if class_type == 'KNN_class':
            self.param_grid = {
                'n_neighbors': list(range(1, 51, 2)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
            self.clf = KNeighborsClassifier()

        elif class_type == 'DT_class':
            # self.param_grid = {
            #     'criterion': ['gini', 'entropy'],
            #     'min_samples_split': [5, 10],
            #     'min_samples_leaf': [1, 2, 5],
            #     'max_depth': [5, 10, 15],
            #     'ccp_alpha': [0.0, 0.001, 0.005, 0.01],
            #     'max_features': [None, 'sqrt', 'log2']
            #
            # }
            self.param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],  # add log_loss for prob. splits
                'max_depth': [3, 5, 10, 20, None],  # shallow → full depth
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'ccp_alpha': [0.0, 0.0001, 0.001, 0.005, 0.01],  # pruning strength
                'max_features': [None, 'sqrt', 'log2']
            }
            self.clf = DecisionTreeClassifier(random_state=42)

        elif class_type == 'RF_class':
            if unbalanced_type == 'None':
                # self.param_grid = {
                    # 'n_estimators': [100, 200, 300],  # Number of trees
                    # 'max_depth': [10, 20, None],  # Tree depth
                    # 'min_samples_split': [2, 5, 10],  # Split threshold
                    # 'min_samples_leaf': [1, 2, 4],  # Leaf size
                    # 'max_features': ['sqrt', 'log2'],  # Feature sampling
                    # 'bootstrap': [True],  # Bootstrapped sampling
                    # 'oob_score': [True],  # Out-of-bag validation
                    # 'max_samples': [0.8, 1.0],  # Fraction of samples per tree
                    # 'class_weight': ['balanced', 'balanced_subsample']  # Handle imbalance internally
                # }
                self.param_grid = {
                    'n_estimators': [200, 500, 1000, 1500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True],
                    'oob_score': [True],
                    'max_samples': [0.7, 0.8, 0.9, 1.0],
                    'class_weight': ['balanced', 'balanced_subsample']
                }

            # else:
            #     self.param_grid = {
            #         'n_estimators': [100, 200, 300],
            #         'max_depth': [10, 20, None],
            #         'min_samples_split': [2, 5, 10],
            #         'min_samples_leaf': [1, 2, 4],
            #         'max_features': ['sqrt', 'log2'],
            #         'bootstrap': [True],
            #         'oob_score': [True],
            #         'max_samples': [0.8, 1.0]
                    # No class_weight here — assume you're using SMOTE/ADASYN
                # }

            self.clf = RandomForestClassifier(random_state=42)


        elif class_type == 'GBM_class':
            self.param_grid = {
                'n_estimators': [200, 500, 1000, 2000],  # boosting rounds
                'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
                'max_depth': [2, 3, 4, 5],
                'subsample': [0.6, 0.8, 1.0],  # stochastic vs. deterministic
                'min_samples_leaf': [1, 5, 10, 20],
                'max_features': ['sqrt', 'log2', None]
            }
            # self.param_grid = {
            #     'n_estimators': [100, 200],
            #     'learning_rate': [0.05],
            #     'max_depth': [3, 4],
            #     'min_samples_split': [2, 10],
            #     'min_samples_leaf': [3, 5],
            #     'subsample': [0.6, 0.8],
            #     'max_features': ['sqrt', 'log2', 0.8]
            # }
            self.clf = GradientBoostingClassifier(random_state=42)

        elif class_type == 'XGBoost_class':
            # self.param_grid = {
            #     'n_estimators': [100, 200],
            #     'learning_rate': [0.05, 0.01, 0.1],
            #     'max_depth': [3, 5, 7],
            #     'subsample': [0.8],
            #     'colsample_bytree': [0.8],
            #     'min_child_weight': [1],
            #     'gamma': [0, 0.1],
            #     'reg_alpha': [0, 0.001],
            #     'reg_lambda': [1]
            # }
            # self.param_grid  = {
            #     'n_estimators': [500, 1000, 2000],
            #     'learning_rate': [0.01, 0.02, 0.05],
            #     'max_depth': [1, 2, 3],
            #     'min_child_weight': [1, 5, 10],
            #     'gamma': [0, 0.1, 0.5],
            #     'subsample': [0.7, 0.9, 1.0],
            #     'colsample_bytree': [0.7, 1.0],
            #     'reg_alpha': [0, 1e-3, 1e-2],
            #     'reg_lambda': [0.5, 1, 2],
            #     # set per fold, not in grid:
            #     # 'scale_pos_weight': [neg_pos_ratio],
            #     # 'monotone_constraints': [(1,)],
            #     # 'tree_method': ['hist']
            # }
            self.param_grid = {
                'n_estimators': [1200, 1600],
                'learning_rate': [0.01, 0.02],
                'max_depth': [2],
                'min_child_weight': [5, 10, 20],
                'gamma': [0.1, 0.3],
                'subsample': [0.75, 0.9],
                'colsample_bytree': [0.6, 0.8],
                'colsample_bylevel': [0.8],  # keep fixed for stability
                'colsample_bynode': [0.8],  # keep fixed for stability
                'reg_alpha': [0.001, 0.01, 0.1],
                'reg_lambda': [2.0, 4.0],
                'max_delta_step': [1],
                'tree_method': ['hist'],
                'n_jobs': [-1]
            }

            self.clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss',objective='binary:logistic' )


        elif class_type == 'LogisticRegression_class':

            self.clf = Pipeline([
                ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                ('clf', LogisticRegression(random_state=42))
            ])

            self.param_grid = [
                # ---- L2 (ridge) ----
                {
                    'clf__penalty': ['l2'],
                    'clf__solver': ['lbfgs', 'newton-cg', 'liblinear', 'saga'],  # all support L2
                    'clf__C': [0.01, 0.1, 1, 10],
                    'clf__class_weight': ['balanced'],
                    'clf__max_iter': [2000],
                    'clf__tol': [1e-3],

                },
                # ---- L1 (lasso) ----
                {
                    'clf__penalty': ['l1'],
                    'clf__solver': ['liblinear', 'saga'],  # only these support L1
                    'clf__C': [0.01, 0.1, 1, 10],
                    'clf__class_weight': ['balanced'],
                    'clf__max_iter': [3000],  # SAGA/L1 may need more iters
                    'clf__tol': [1e-3],
                },

                {
                    'clf__penalty': ['elasticnet'],
                    'clf__solver': ['saga'],  # ONLY saga supports elastic-net
                    'clf__l1_ratio': [0.1, 0.5, 0.9],  # mix of L1/L2
                    'clf__C': [0.01, 0.1, 1, 10],
                    'clf__class_weight': ['balanced'],
                    'clf__max_iter': [5000],  # more iters for convergence
                    'clf__tol': [1e-3],
                },
            ]

        else:
            raise ValueError(f"Unknown classifier type: {class_type}")

        self.param_combinations = [dict(zip(self.param_grid.keys(), v))
                                   for v in product(*self.param_grid.values())]
