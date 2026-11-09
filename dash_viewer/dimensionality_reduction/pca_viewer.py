import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from kneed import KneeLocator

def calculate_pca(n_components, array_curves):
    """Fit PCA and return (scores, model, components used).

    n_components is the slider value: 0 means "however many reach 95% variance",
    100 means "all of them", anything else is taken literally.
    """
    n_samples, n_features = array_curves.shape
    max_components = min(n_samples, n_features)

    pca_full = PCA().fit(array_curves)

    var3 = variance_for_k(pca_full, k=3)
    print(f'First 3 PCs explain {var3 * 100:.2f}% of the variance.')

    # Opt-in only: plt.show() needs the main thread, and this runs on a Dash
    # callback worker. Set CARDIO_PCA_PLOTS=1 when calling from a script.
    if os.environ.get('CARDIO_PCA_PLOTS') == '1':
        plot_cum_variance_with_k(pca_full, k=3, threshold=0.95)

    elbow_idx, optimal_components = plot_explained_variance(pca_full)
    print(f'Knee of the variance curve at {elbow_idx} PCs; '
          f'95% reached at {optimal_components} PCs.')

    if n_components == 0:
        n_components_used = optimal_components
    elif n_components == 100:
        n_components_used = max_components
    else:
        # Honour the request. It used to be capped at optimal_components, which
        # silently handed back a different number than the slider showed.
        n_components_used = max(1, min(n_components, max_components))
        if n_components > max_components:
            print(f'[INFO] {n_components} components requested, only {max_components} exist; '
                  f'using {max_components}.')
        elif n_components_used > optimal_components:
            print(f'[INFO] using {n_components_used} components; '
                  f'95% of the variance is already reached at {optimal_components}.')

    # Truncating the fit already computed avoids a second full SVD - which in
    # segment mode ran once per segment.
    pca_var = truncate_pca(pca_full, n_components_used)
    pca_result = pca_var.transform(array_curves)

    explained_variance_n = np.sum(pca_var.explained_variance_ratio_)
    print(f"Cumulative explained variance with {n_components_used} PCs: {explained_variance_n:.4f}")

    return pca_result, pca_var, n_components_used


def truncate_pca(pca_full, n_components):
    """A PCA model keeping only the first n_components of an existing fit.

    Equivalent to refitting PCA(n_components=k) on the same data, without
    redoing the decomposition.
    """
    truncated = copy.deepcopy(pca_full)
    truncated.components_ = pca_full.components_[:n_components]
    truncated.explained_variance_ = pca_full.explained_variance_[:n_components]
    truncated.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    truncated.singular_values_ = pca_full.singular_values_[:n_components]
    truncated.n_components_ = n_components
    truncated.n_components = n_components
    return truncated


def r2_score_reconstruction(pca, curves):
    """Fraction of variance the reconstruction accounts for: 1 - MSE / Var(X).

    The same measure the autoencoder reports, so the two are comparable at equal
    latent dimension.
    """
    reconstructed = pca.inverse_transform(pca.transform(curves))
    residual = float(np.mean((curves - reconstructed) ** 2))
    total = float(np.mean((curves - curves.mean(axis=0)) ** 2))
    if total == 0:
        return float('nan')
    return 1.0 - residual / total


def holdout_r2(array_curves, n_components, val_fraction=0.15, random_state=42):
    """Fit on a training split, score the studies held out of it.

    explained_variance_ratio_ is measured on the data the basis was fitted to,
    so it says nothing about how the basis handles an unseen study. This does.
    Returns (train_r2, holdout_r2, n_components_used) or None when the cohort is
    too small to split.
    """
    n_samples = len(array_curves)
    n_val = int(round(n_samples * val_fraction))
    if val_fraction <= 0 or n_val < 1 or n_samples - n_val < 2:
        return None

    order = np.random.default_rng(random_state).permutation(n_samples)
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_curves, val_curves = array_curves[train_idx], array_curves[val_idx]

    k = max(1, min(n_components, min(len(train_curves), array_curves.shape[1])))
    pca = PCA(n_components=k, svd_solver='full').fit(train_curves)

    return r2_score_reconstruction(pca, train_curves), r2_score_reconstruction(pca, val_curves), k


def plot_explained_variance(pca, threshold=0.95):
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = len(explained_variance)

    # kl = KneeLocator(range(1, num_components + 1), explained_variance, curve="convex", direction="decreasing")
    kl = KneeLocator(range(1, num_components + 1), explained_variance, curve="concave", direction="increasing")

    elbow_idx = kl.elbow if kl.elbow else 1

    optimal_components = np.argmax(explained_variance >= threshold) + 1

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, num_components + 1), explained_variance, marker='o', linestyle='--', label="Cumulative Variance")
    #
    # plt.axvline(x=elbow_idx, color='red', linestyle='--', label=f"Elbow Point ({elbow_idx} PCs)")
    # plt.scatter(elbow_idx, explained_variance[elbow_idx - 1], color='red', s=100, label="Elbow")
    #
    # plt.axvline(x=optimal_components, color='blue', linestyle=':', label=f"95% Variance ({optimal_components} PCs)")
    # plt.scatter(optimal_components, explained_variance[optimal_components - 1], color='blue', s=100,
    #             label="95% Variance")
    #
    # plt.xlabel("Number of Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.title("Explained Variance vs. Number of Components")
    # plt.legend()
    # plt.grid(True)
    #
    # plt.show()

    return elbow_idx, optimal_components


def reconstruct_traces_pca(pca_result, pca_var):
    reconstructed_curves = pca_var.inverse_transform(pca_result)

    return reconstructed_curves

def variance_for_k(pca, k=3):
    """Return cumulative explained variance for the first k PCs (0–1 scale)."""
    k = min(k, len(pca.explained_variance_ratio_))
    return float(np.sum(pca.explained_variance_ratio_[:k]))


def plot_cum_variance_with_k(pca, k=3, threshold=0.95):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k = min(k, len(evr))

    plt.figure(figsize=(7,4.5))
    plt.plot(range(1, len(cum)+1), cum, marker='o', linestyle='--', label='Cumulative variance')
    # mark k PCs
    plt.axvline(k, linestyle='--', label=f'{k} PCs')
    plt.scatter(k, cum[k-1], s=80)
    plt.text(k+0.3, cum[k-1], f'{cum[k-1]*100:.1f}%', va='center')

    # mark 95% ref
    opt_k = int(np.argmax(cum >= threshold) + 1)
    plt.axvline(opt_k, linestyle=':', label=f'{int(threshold*100)}% at {opt_k} PCs')

    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Explained variance vs. number of PCs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
