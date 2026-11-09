from collections import OrderedDict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (ParameterGrid, ParameterSampler,
                                     StratifiedKFold)


# Exhaustive search over these grids is 200-6144 fits per CV fold. With nested CV
# that becomes intractable, so grids larger than this are randomly subsampled.
# Set n_iter=None in get_param_grid() to force the full exhaustive grid.
DEFAULT_SEARCH_BUDGET = 60


# Single source of truth for "which classifiers exist and what are they called".
# An OrderedDict of class_type -> human label; iteration order is the order the
# UI should list them in. The viewer and the model-comparison view both read
# this instead of hard-coding their own copy of the list.
CLASSIFIER_LABELS = OrderedDict([
    ('KNN_class', 'KNN'),
    ('DT_class', 'Decision Tree'),
    ('RF_class', 'Random Forest'),
    ('GBM_class', 'Gradient Boosting'),
    ('XGBoost_class', 'XGBoost'),
    ('LogisticRegression_class', 'Logistic Regression'),
    ('Champion_class', 'Champion (calibrated XGBoost)'),
    ('Stacking_class', 'Stacking Ensemble'),
])

# The subset worth running in an "all models" comparison sweep. KNN is excluded:
# it is commented out of the current viewer dropdown, and a distance-based model
# on ~252 exams of high-dimensional strain traces is not a serious contender.
COMPARISON_DEFAULT_MODELS = [
    'DT_class',
    'RF_class',
    'GBM_class',
    'XGBoost_class',
    'LogisticRegression_class',
    'Champion_class',
    'Stacking_class',
]


class ModelParams:
    def __init__(self):
        self.clf = None
        self.param_grid = None
        self.param_combinations = None

    def get_param_grid(self, class_type, unbalanced_type, n_iter=DEFAULT_SEARCH_BUDGET, random_state=42):
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
            self.param_grid = {
                'n_estimators': [200, 500, 1000, 1500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True],
                'max_samples': [0.7, 0.8, 0.9, 1.0],
                # Correct the imbalance once. With a sampler in the pipeline the
                # classes are already balanced by the time the forest sees them,
                # and class_weight on top over-predicts the minority class.
                'class_weight': (['balanced', 'balanced_subsample']
                                 if unbalanced_type == 'None' else [None]),
            }
            # oob_score was in the grid but nothing ever read oob_score_; it made
            # every fit compute out-of-bag predictions for nothing.
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

            # eval_metric was 'mlogloss' (multiclass) against a binary objective;
            # 'aucpr' matches the PR-AUC this pipeline selects on. use_label_encoder
            # was removed in xgboost 2.0 and is no longer passed.
            self.clf = XGBClassifier(random_state=42, eval_metric='aucpr',
                                     objective='binary:logistic')


        elif class_type == 'LogisticRegression_class':
            # See the RF note: balance once, either here or with a sampler.
            weighting = ['balanced'] if unbalanced_type == 'None' else [None]

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
                    'clf__class_weight': weighting,
                    'clf__max_iter': [2000],
                    'clf__tol': [1e-3],

                },
                # ---- L1 (lasso) ----
                {
                    'clf__penalty': ['l1'],
                    'clf__solver': ['liblinear', 'saga'],  # only these support L1
                    'clf__C': [0.01, 0.1, 1, 10],
                    'clf__class_weight': weighting,
                    'clf__max_iter': [3000],  # SAGA/L1 may need more iters
                    'clf__tol': [1e-3],
                },

                {
                    'clf__penalty': ['elasticnet'],
                    'clf__solver': ['saga'],  # ONLY saga supports elastic-net
                    'clf__l1_ratio': [0.1, 0.5, 0.9],  # mix of L1/L2
                    'clf__C': [0.01, 0.1, 1, 10],
                    'clf__class_weight': weighting,
                    'clf__max_iter': [5000],  # more iters for convergence
                    'clf__tol': [1e-3],
                },
            ]

        elif class_type == 'Champion_class':
            # The model this project defends: a shallow, heavily regularised
            # gradient booster wrapped in probability calibration.
            #
            # Calibration: CalibratedClassifierCV with method='sigmoid' (Platt).
            # Isotonic regression is non-parametric and needs far more data than
            # the ~252 exams / minority-class handful available here - it would
            # fit the calibration curve to noise. Platt scaling has two
            # parameters and degrades gracefully. cv=3 keeps each internal
            # calibration split large enough to still contain positives; those
            # splits are taken from the training fold only, so nothing leaks.
            base = XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr',
                tree_method='hist',
                random_state=42,
                n_jobs=-1,
            )
            self.clf = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3)

            # Grid keys must address the WRAPPED estimator. The surrounding
            # imbalanced-learn pipeline sets params on its 'clf' step via
            # clone(clf).set_params(**combo), and clf here is the calibrator, so
            # every tuned XGBoost parameter carries the 'estimator__' prefix.
            #
            # DO NOT ADD 'estimator__scale_pos_weight' HERE. It is set per fold
            # from that training fold's own negative/positive ratio; computing it
            # from the whole cohort leaks the held-out class balance into the fit.
            #
            # 2^8 = 256 combinations before sampling - deliberately tight so the
            # champion can be re-run often.
            self.param_grid = {
                'estimator__max_depth': [2, 3],              # shallow trees only
                'estimator__learning_rate': [0.01, 0.03],    # slow, many rounds
                'estimator__n_estimators': [800, 1500],
                'estimator__min_child_weight': [5, 10],      # no leaves on 1-2 cases
                'estimator__reg_alpha': [0.01, 0.1],         # L1
                'estimator__reg_lambda': [2.0, 5.0],         # L2
                'estimator__colsample_bytree': [0.6, 0.8],   # column subsampling
                'estimator__max_delta_step': [1, 3],         # stabilises the logistic
                                                             # updates under imbalance
                'estimator__subsample': [0.8],               # row subsampling, fixed
                'estimator__gamma': [0.1],                   # fixed split penalty
            }

        elif class_type == 'Stacking_class':
            # Same balancing rule as RF / LogisticRegression above: when a sampler
            # is already balancing the data upstream, class_weight on the base
            # learners corrects the imbalance a second time and over-predicts the
            # minority class. Balance once - here or with the sampler, not both.
            weighting = 'balanced' if unbalanced_type == 'None' else None

            base_estimators = [
                ('lr', LogisticRegression(penalty='l2', solver='lbfgs', C=1.0,
                                          max_iter=2000, tol=1e-3,
                                          class_weight=weighting, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                              max_features='sqrt',
                                              class_weight=weighting,
                                              random_state=42, n_jobs=-1)),
                # XGBoost has no class_weight; its imbalance knob is
                # scale_pos_weight, which is set per fold outside this file and
                # so must not appear in the grid (see Champion_class).
                ('xgb', XGBClassifier(objective='binary:logistic',
                                      eval_metric='aucpr', tree_method='hist',
                                      n_estimators=600, max_depth=2,
                                      min_child_weight=5, subsample=0.8,
                                      colsample_bytree=0.8, reg_lambda=2.0,
                                      max_delta_step=1, random_state=42, n_jobs=-1)),
            ]

            self.clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(penalty='l2', solver='lbfgs',
                                                   max_iter=2000, tol=1e-3,
                                                   class_weight=weighting,
                                                   random_state=42),
                stack_method='predict_proba',
                passthrough=False,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                n_jobs=1,
            )

            # Stacking refits every base learner on every internal split, so one
            # candidate already costs ~3x(LR+RF+XGB). Tune three things only:
            # 3*3*2 = 18 combinations.
            self.param_grid = {
                'final_estimator__C': [0.01, 0.1, 1.0],
                'rf__max_depth': [3, 5, None],
                'xgb__learning_rate': [0.02, 0.05],
            }

        else:
            raise ValueError(f"Unknown classifier type: {class_type}")

        # param_grid is a dict for most models but a list of dicts for
        # LogisticRegression, where penalty/solver combinations are not a cross
        # product. ParameterGrid handles both; the old zip(product(...)) call
        # raised AttributeError on the list form, which the caller silently
        # swallowed, so LogisticRegression never actually ran.
        full_grid = ParameterGrid(self.param_grid)

        if n_iter is not None and len(full_grid) > n_iter:
            self.param_combinations = list(
                ParameterSampler(self.param_grid, n_iter=n_iter, random_state=random_state)
            )
        else:
            self.param_combinations = list(full_grid)
