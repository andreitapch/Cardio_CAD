import numpy as np
from sklearn.decomposition import PCA


def calculate_pca(n_components, array_curves):
    if n_components == 1:
        n_components = None
    pca_var = PCA(n_components=n_components, svd_solver='full')
    pca_result = pca_var.fit_transform(array_curves)
    return pca_result, pca_var


def reconstruct_traces_pca(pca_result, pca_var):
    reconstructed_curves = pca_var.inverse_transform(pca_result)
    return reconstructed_curves

