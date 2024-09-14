import os

from pca_strain import PCAStrain
# from create_plots import plot_curves, plot_pca, plot_principal_components
# from curve_labeler import CurveLabeler
# from knn_classifier import KNNClassifier

from interpolate_curves_sm import InterpolateCurves_sm
from interpolate_curves import InterpolateCurves


def temporal_alignment(root_path, segments, phases, ref_label):
    return InterpolateCurves_sm(root_path, segments, phases, ref_label)
    # return InterpolateCurves(root_path, segments, phases, ref_label)


# def label_curves(curves):
#     interp_strain_labelled = CurveLabeler(curves)
#     return curves


def run_PCA(curves,name):
    principal_components = PCAStrain(curves, name)
    return principal_components


# def curve_classification(curves, principal_components, to_predict):
#     predictions = KNNClassifier(principal_components, to_predict)
#
#     return predictions


class MainClass:
    def __init__(self, root_path, segments, phases, ref_label, plot_curves_bol):
        self.interp_sr_pca = None
        self.interp_strain_pca = None
        self.predictions = None
        self.labels = None
        self.principal_components = None

        self.curves = temporal_alignment(root_path, segments, phases, ref_label)
        # self.labelling = label_curves(self.curves)  # TODO Doing labeling
        # self.plot_strains(plot_curves_bol)

        self.pca = run_PCA(self.curves.interpolated_strain,'Strain ')  # TODO - Labeling??
        self.interp_sr_pca = self.interp_strain_pcaT = run_PCA(self.curves.interpolated_sr,'Strain Rate ')  # TODO - Labeling??

        # self.curve_classification(self, to_predict=1) #TODO

    # def plot_strains(self, plot_curves_boolean):
    #     if plot_curves_boolean:
    #         plot_curves(self.curves.strain_sorted, self.curves.label_sorted,
    #                     self.curves.ev_time_sorted, self.curves.length,
    #                     'Strain-Curves')
    #
    #         plot_curves(self.curves.sr_sorted, self.curves.label_sorted,
    #                     self.curves.ev_time_sorted, self.curves.length,
    #                     'SR-Curves')
    #
    #         plot_curves(self.curves.interpolated_strain, self.curves.label_sorted,
    #                     self.curves.ev_time_sorted, self.curves.length,
    #                     'Interpolated-Strain-Curves')
    #
    #         plot_curves(self.curves.interpolated_sr, self.curves.label_sorted,
    #                     self.curves.ev_time_sorted, self.curves.length,
    #                     'Interpolated-SR-Curves')


if __name__ == "__main__":
    root_path = os.path.dirname('D:/PHD/Experiments/CAD/Data-strain/')
    plot_curves_boolean = False
    segments = 18
    phases = 6
    ref_label = [1, 2, 3, 4, 5, 6]

    main = MainClass(root_path, segments, phases, ref_label, plot_curves_boolean)
