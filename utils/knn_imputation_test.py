import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Step 1: Create a sample dataset with missing values
X = np.array([
    [1.0, 2.0, np.nan],
    [3.0, 4.0, 5.0],
    [6.0, np.nan, 8.0],
    [9.0, 10.0, 11.0]
])

print("Original Data Matrix with Missing Values:")
print(X)

# Step 2: Define the KNN imputation function
def knn_imputation(X, k=3, epsilon=1e-5):
    nan_mask = np.isnan(X)
    X_imputed = X.copy()
    nan_replacement = np.nanmean(X, axis=0)
    X_imputed[nan_mask] = np.take(nan_replacement, np.where(nan_mask)[1])

    for i in range(X.shape[0]):
        if np.any(nan_mask[i]):
            non_missing_mask = ~nan_mask[i]
            X_i = X[i, non_missing_mask].reshape(1, -1)
            X_complete = X_imputed[:, non_missing_mask]

            distances = pairwise_distances(X_i, X_complete, metric='euclidean').flatten()
            nearest_indices = np.argsort(distances)[:k]

            di = distances[nearest_indices]
            weights = 1 / (di + epsilon)
            weights /= weights.sum()

            for j in np.where(nan_mask[i])[0]:
                X_imputed[i, j] = np.dot(weights, X_imputed[nearest_indices, j])

    return X_imputed

# Step 3: Define the cross-validation function
def cross_validate_knn_imputation(X, k=3, n_splits=5):
    kf = KFold(n_splits=n_splits)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        X_train_imputed = knn_imputation(X_train, k=k)

        nan_mask_test = np.isnan(X_test)
        X_test_imputed = X_test.copy()
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                if nan_mask_test[i, j]:
                    non_missing_mask = ~np.isnan(X_test[i])
                    X_i = X_test[i, non_missing_mask].reshape(1, -1)
                    X_complete = X_train_imputed[:, non_missing_mask]

                    distances = pairwise_distances(X_i, X_complete, metric='euclidean').flatten()
                    nearest_indices = np.argsort(distances)[:k]

                    di = distances[nearest_indices]
                    weights = 1 / (di + epsilon)
                    weights /= weights.sum()

                    X_test_imputed[i, j] = np.dot(weights, X_train_imputed[nearest_indices, j])

        mse = mean_squared_error(X_test[~nan_mask_test], X_test_imputed[~nan_mask_test])
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

# Step 4: Implement the grid search for the best k
def find_best_k(X, k_values, n_splits=5):
    best_k = None
    best_mse = float('inf')

    for k in k_values:
        mean_mse, _ = cross_validate_knn_imputation(X, k=k, n_splits=n_splits)
        print(f'k={k}, Mean MSE={mean_mse}')

        if mean_mse < best_mse:
            best_mse = mean_mse
            best_k = k

    return best_k, best_mse

# Step 5: Define the range of k values to test
k_values = [1, 2, 3, 4, 5]

# Step 6: Find the best k
best_k, best_mse = find_best_k(X, k_values, n_splits=3)

print("Best k:", best_k)
print("Best Mean MSE:", best_mse)
