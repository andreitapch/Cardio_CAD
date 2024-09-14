import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from create_plots import plot_pca, plot_principal_components


class PCAStrain:
    def __init__(self, curves,name):
        self.principalComponents = []
        self.perform_pca(curves,name)


    @staticmethod
    def perform_pca(curve,name):
        curve = np.array(curve)
        reshaped_curve = curve.reshape(len(curve), -1)

        pca = PCA()
        principal_components = pca.fit_transform(reshaped_curve)
        pca_transpose = principal_components.transpose((1, 0))

        plot_principal_components(principal_components)
        plot_principal_components(pca_transpose)

        return principal_components, pca_transpose
