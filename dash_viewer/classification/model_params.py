from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


class ModelParams:
    def get_param_grid(self, class_type):
        if class_type == 'KNN_class':
            self.param_grid = {
                'n_neighbors': list(range(1, 31)),
                'weights': ['uniform', 'distance'],
                'metric': ['braycurtis', 'euclidean']  # 'cosine', 'manhattan', 'chebyshev', 'minkowski',
            }
            self.clf = KNeighborsClassifier()
        elif class_type == 'DT_class':
            self.param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],
                # 'splitter': ['best'], #, 'random' for more than one feature
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5, 10],
                # 'class_weight': [None, 'balanced', {0: 1, 1: 10}, {0: 1, 1: 20}],
                # 'min_impurity_decrease': [0.0, 0.01, 0.1],
                'ccp_alpha':
                    [0.0, 0.01, 0.1]
            }
            self.clf = DecisionTreeClassifier(random_state=42)
        elif class_type == 'RF_class':
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [4, 5, 6, 7, 8],
                'criterion': ['gini', 'entropy'],
                'class_weight': [None, 'balanced']
            }
            self.clf = RandomForestClassifier(random_state=42)
        elif class_type == 'GBM_class':
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.05],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            }
            self.clf = GradientBoostingClassifier(random_state=42)
        elif class_type == 'XGBoost_class':
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.05],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'scale_pos_weight': [1, 10, 25, 50, 75, 100]
            }
            self.clf = XGBClassifier(random_state=42, use_label_encoder=False,
                                     eval_metric='mlogloss')
        elif class_type == 'LightGBM_class':
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.05],
                'max_depth': [-1, 3, 4, 5, 6, 7],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'class_weight': [None, 'balanced']
            }
            self.clf = LGBMClassifier(random_state=42)
        elif class_type == 'CatBoost_class':
            self.param_grid = {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.05],
                'depth': [3, 4, 5, 6, 7],
                'l2_leaf_reg': [1, 3, 5, 7],
                'scale_pos_weight': [1, 10, 25, 50, 75, 100]
            }
            self.clf = CatBoostClassifier(random_state=42, verbose=0)
        elif class_type == 'SVM_class':
            self.param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }
            self.clf = SVC(probability=True, random_state=42)
        elif class_type == 'LogisticRegression_class':
            self.param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 300],
                'class_weight': [None, 'balanced']
            }
            self.clf = LogisticRegression(random_state=42)
        elif class_type == 'NaiveBayes_class':
            self.param_grid = {
                'alpha': [0.01, 0.1, 1, 10, 100]
            }
            self.clf = MultinomialNB()  # Choose MultinomialNB for this example, GaussianNB and BernoulliNB can also be used
        else:
            raise ValueError(f"Unknown classifier type: {class_type}")
