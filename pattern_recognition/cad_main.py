# TODO I should convert this to CheckBox in Welcome Menu in dash
from pattern_recognition.data.SPEQLE_data.source_speqle import SpeqleDataProcessor
from pattern_recognition.data.speckle_tracking.source_speckle_tracking import SpeckleTrackingProcessor

studyType = 'BL'
interpSelection = 'max' # 'avg' OR 'max'
concat_traces_by = 'full_traces'  # 'full_traces' OR 'segment'
selected_source_data = 'Philips'  # 'SPEQLE' OR 1 'ST' OR 2 'Philips' as interp is based on GE
read_traces = True
interp_traces_db = True
calculate_pca = False
plot_curves = False
plot_interp_curves = False
plot_PCA_curves = False


def main_process():
    global speqle_processor
    if selected_source_data == 'ST':
        speqle_processor = SpeckleTrackingProcessor(studyType, selected_source_data, concat_traces_by, interpSelection, read_traces, interp_traces_db)
        speqle_processor.process()
    elif selected_source_data == 'Philips':
        speqle_processor = SpeckleTrackingProcessor(studyType, selected_source_data, concat_traces_by, interpSelection,
                                                    read_traces, interp_traces_db)
        speqle_processor.process()
    elif selected_source_data == 'SPEQLE':
        speqle_processor = SpeqleDataProcessor(studyType, concat_traces_by, interpSelection, read_traces, plot_curves,
                                               interp_traces_db, plot_interp_curves, calculate_pca, plot_PCA_curves)
        speqle_processor.process()

    return speqle_processor


speqle_processor = main_process()
