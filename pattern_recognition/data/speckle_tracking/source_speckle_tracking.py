from pattern_recognition.strain_analysis.align_strain_curves import AlignStrainTraces
from pattern_recognition.data.speckle_tracking.rename_ste_folders import RenameSTEFolders
from pattern_recognition.data.speckle_tracking.listFilesToRead import ListAvailableStFiles
from pattern_recognition.data.speckle_tracking.listPhiFilesToRead import ListAvailablePhiFiles
from pattern_recognition.data.speckle_tracking.read_st_file_traces import StrainTraceFileReader
from pattern_recognition.data.speckle_tracking.read_philips_file_traces import PhilipsFileReader


class SpeckleTrackingProcessor:
    def __init__(self, studyType, selected_source_data, concat_traces_by, interpSelection, read_traces,
                 interp_traces_db):
        self.studyType = studyType
        self.selected_source_data = selected_source_data
        self.concat_traces_by = concat_traces_by
        self.interpSelection = interpSelection
        self.rename_folders = False
        self.read_traces = read_traces
        self.interp_traces_db = interp_traces_db
        self.calculate_pca = False
        self.plot_curves = False
        self.plot_interp_curves = False
        self.speqle_traces = None
        self.cardiac_eventDF = None
        self.Cardiac_events_order = ['AVC']

    def process(self):
        if self.read_traces:  # read traces from different sources

            if self.selected_source_data == 'ST':
                if self.rename_folders:  # First rename speckle tracking folders
                    RenameSTEFolders()
                list_st_files = ListAvailableStFiles()
                StrainTraceFileReader(list_st_files, self.studyType)

            elif self.selected_source_data == 'Philips':
                list_st_files = ListAvailablePhiFiles()
                PhilipsFileReader(list_st_files, self.studyType)
        if self.plot_curves:
            pass

        # Align Strain Curves and interpolate
        if self.interp_traces_db:
            AlignStrainTraces(self.selected_source_data, self.interpSelection)
            print('Interpolation finished')
