import pandas as pd

from pattern_recognition.utils.utils import event_loader
from pattern_recognition.utils.parameters_dics import cardiac_events_order
from pattern_recognition.utils.plot.trace_plotter import TracePlotter
from pattern_recognition.ml_algorithms.calculate_PCA import PCAStrainTraces
from pattern_recognition.data.SPEQLE_data.load_speqle_files import SpeqleFileLoader
from pattern_recognition.data_preprocessing.speqle.temporalAligment import TemporalAlignment
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data_preprocessing.speqle.concatenate_segments import SegmentsConcatenator
from pattern_recognition.utils.plot_dics import wall_color_dict, segment_color_dict, Cardiac_view_dict
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle


class SpeqleDataProcessor:
    def __init__(self, studyType, concat_traces_by, interpSelection, read_traces, plot_curves,
                 interp_traces_db, plot_interp_curves, calculate_pca, plot_PCA_curves):

        self.tracesDF = None
        self.pca_result = None
        self.calculate_PCA = None
        self.reconstructed_traces = None
        self.read_traces = read_traces
        self.plot_curves = plot_curves
        self.calculate_pca = calculate_pca
        self.plot_interp_curves = plot_interp_curves
        self.plot_PCA_curves = plot_PCA_curves

        self.interp_traces_db = interp_traces_db
        self.studyType = studyType
        self.concat_traces_by = concat_traces_by
        self.interpSelection = interpSelection
        self.Cardiac_events_order = cardiac_events_order

    def plot_traces(self, table_name, isInterp, cardiac_eventT, isPCA):
        cardiac_evDF = None
        if cardiac_eventT is not None:
            cardiac_eventDF = load_traces_speqle(self.studyType, cardiac_eventT,None)
            cardiac_evDF, filtered_cardiac_ev = event_loader(
                self.Cardiac_events_order, cardiac_eventDF.tracesDF, self.interpSelection)

        if not isPCA:
            speqle_traces = load_traces_speqle(self.studyType, table_name, None)
            tracesDF = speqle_traces.tracesDF
        else:
            tracesDF = self.reconstructed_traces
        trace_plotter = TracePlotter(tracesDF, cardiac_evDF, self.concat_traces_by)

        trace_plotter.group_traces('segment_wall')
        trace_plotter.plot_trace('Wall', wall_color_dict, isInterp, 1200,700, True)

        trace_plotter.group_traces('curve_segment')
        trace_plotter.plot_trace('Segment', segment_color_dict, isInterp, 2000, 700, True)

        trace_plotter.group_traces('cardiac_view')
        trace_plotter.plot_trace('Cardiac View', Cardiac_view_dict, isInterp,1000, 700, True )

    def process(self):
        if self.read_traces:
           SpeqleFileLoader(self.studyType)

        if self.plot_curves:
            self.plot_traces('speqle_segmentalStrain_T', 'Original',None, False)

        # Align Strain Curves and interpolate
        if self.interp_traces_db:
            seg_strain_tracesDF = load_traces_speqle(self.studyType, 'speqle_segmentalStrain_T', None)
            seg_sr_tracesDF = load_traces_speqle(self.studyType, 'speqle_segmentalSr_T', None)
            cardiac_eventDF = load_traces_speqle(self.studyType, 'speqle_cardiacEvent_T', None)
            strain_tracesDF = load_traces_speqle(self.studyType, 'speqle_strainValues_T', None)

            interp_strain = TemporalAlignment(cardiac_eventDF.tracesDF, seg_strain_tracesDF.tracesDF, seg_sr_tracesDF.tracesDF,
                                              strain_tracesDF.tracesDF, self.interpSelection)

            ## TODO delete 0.00

        # Plot Interpolated Strain Curves
        if self.plot_interp_curves:
            self.plot_traces('speqle_segmentalStrainInterp_T', 'Interpolated','speqle_cardiacEvent_T', False)

        # # Calculate PCA
        # if self.calculate_pca:
        #     seg_strain_interpDF = load_traces_speqle(self.studyType, 'speqle_segmentalStrainInterp_T', None)
        #     if self.concat_traces_by == 'full_traces':
        #         concatenated_strain = CurveConcatenator(seg_strain_interpDF.tracesDF, isReconstructed=False)
        #         self.tracesDF = concatenated_strain.concatenate_traces()
        #     elif self.concat_traces_by == 'segment':
        #         concatenated_strain = SegmentsConcatenator(seg_strain_interpDF.tracesDF)
        #         self.tracesDF = concatenated_strain.concatenate_segment_traces()
        #
        #     print('Performing PCA ')
        #     self.calculate_PCA = PCAStrainTraces(self.tracesDF, self.studyType, self.concat_traces_by)
        #     if self.concat_traces_by == 'full_traces':
        #         self.calculate_PCA.prepare_all_traces()
        #         self.pca_result = self.calculate_PCA.calculate_pca(None)
        #         self.calculate_PCA.reconstruct_traces(None)
        #         self.reconstructed_traces = self.calculate_PCA.PCA_tracesDF
        #
        #     elif self.concat_traces_by == 'segment':
        #         self.calculate_PCA.prepare_segment_traces()
        #         self.reconstructed_traces = self.calculate_PCA.reconstructed_tracesDF
        #
        #     if self.plot_PCA_curves:
        #         self.plot_traces(None,'Reconstructed',  'speqle_cardiacEvent_T', isPCA=True)
