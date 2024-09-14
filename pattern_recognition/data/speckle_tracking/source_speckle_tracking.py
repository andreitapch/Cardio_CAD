from pattern_recognition.utils.plot.trace_plotter import TracePlotter
from pattern_recognition.ml_algorithms.calculate_PCA import PCAStrainTraces
from pattern_recognition.strain_analysis.align_strain_curves import AlignStrainTraces
from pattern_recognition.data.speckle_tracking.rename_ste_folders import RenameSTEFolders
from pattern_recognition.data.speckle_tracking.listFilesToRead import ListAvailableStFiles
from pattern_recognition.data.speckle_tracking.read_st_file_traces import StrainTraceFileReader


class SpeckleTrackingProcessor:
    def __init__(self, studyType='BL', selected_source_data='ST', concat_traces_by='segment',
                 interpSelection='avg'):
        self.studyType = studyType
        self.selected_source_data = selected_source_data
        self.concat_traces_by = concat_traces_by
        self.interpSelection = interpSelection
        self.rename_folders = False
        self.read_traces = False
        self.interp_traces_db = False
        self.calculate_pca = True
        self.plot_curves = False
        self.plot_interp_curves = False
        self.speqle_traces = None
        self.cardiac_eventDF = None
        self.Cardiac_events_order = ['AVC']

    def process(self):
        if self.read_traces:  # read traces from different sources
            if self.rename_folders:  # First rename speckle tracking folders
                RenameSTEFolders()
            list_st_files = ListAvailableStFiles()
            strain_traces = StrainTraceFileReader(list_st_files)

        if self.plot_curves:
            pass

        # Align Strain Curves and interpolate
        if self.interp_traces_db:
            aligned_traces = AlignStrainTraces()

        # Plot Interpolated Strain Curves
        if self.plot_interp_curves:
            pass

        # Calculate PCA
        if self.calculate_pca:
            tracesDF = None
            print('Performing PCA ')
            calculate_PCA = PCAStrainTraces(tracesDF, self.studyType)
            if self.concat_traces_by == 'full_traces':
                calculate_PCA.prepare_all_traces()
                pca_result = calculate_PCA.calculate_pca()
                calculate_PCA.reconstruct_curve()

            elif self.concat_traces_by == 'segment':
                calculate_PCA.prepare_segment_traces()
