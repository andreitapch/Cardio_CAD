import numpy as np
from sklearn.decomposition import PCA

OPTIMAL, MAXIMUM = 0, 100


def components_for(request, full, threshold=0.95):
    cumulative = np.cumsum(full.explained_variance_ratio_)
    if request == OPTIMAL:
        return int(np.argmax(cumulative >= threshold)) + 1
    if request == MAXIMUM:
        return len(cumulative)
    return max(1, min(int(request), len(cumulative)))


def truncate(full, k):
    model = PCA(n_components=k)
    model.components_ = full.components_[:k]
    model.explained_variance_ = full.explained_variance_[:k]
    model.explained_variance_ratio_ = full.explained_variance_ratio_[:k]
    model.singular_values_ = full.singular_values_[:k]
    model.mean_ = full.mean_
    model.n_components_ = k
    model.n_features_in_ = full.n_features_in_
    model.n_samples_ = full.n_samples_
    model.noise_variance_ = full.noise_variance_
    return model


def fit(x, request):
    full = PCA(svd_solver="full").fit(x)
    return truncate(full, components_for(request, full))


def r2(model, x):
    reconstructed = model.inverse_transform(model.transform(x))
    total = float(np.mean((x - x.mean(axis=0)) ** 2))
    return float("nan") if total == 0 else 1.0 - float(np.mean((x - reconstructed) ** 2)) / total


def holdout(x, k, val_index, train_index):
    model = PCA(n_components=min(k, len(train_index), x.shape[1]), svd_solver="full").fit(x[train_index])
    return r2(model, x[train_index]), r2(model, x[val_index])
