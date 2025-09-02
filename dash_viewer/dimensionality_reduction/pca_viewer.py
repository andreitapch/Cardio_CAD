import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from kneed import KneeLocator

def calculate_pca(n_components, array_curves):
    n_samples, n_features = array_curves.shape
    max_components = min(n_samples, n_features)

    pca_full = PCA().fit(array_curves)

    var3 = variance_for_k(pca_full, k=3)
    print(f'First 3 PCs explain {var3 * 100:.2f}% of the variance.')

    plot_cum_variance_with_k(pca_full, k=3, threshold=0.95)

    elbow_idx, optimal_components = plot_explained_variance(pca_full)

    if n_components == 0:
        n_components_used = optimal_components
    elif n_components == 100:
        n_components_used = max_components
    else:
        n_components_used = max(1, min(n_components, optimal_components))

    pca_var = PCA(n_components=n_components_used, svd_solver='full')
    pca_result = pca_var.fit_transform(array_curves)

    explained_variance_n = np.sum(pca_var.explained_variance_ratio_)
    print(f"Cumulative explained variance with {n_components_used} PCs: {explained_variance_n:.4f}")

    return pca_result, pca_var, n_components_used


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
