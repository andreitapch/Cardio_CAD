from data.SPEQLE_data.load_speqle_files import SpeqleFileLoader
from pattern_recognition.data_preprocessing.speqle.concatenate_segments import SegmentsConcatenator
from pattern_recognition.ml_algorithms.calculate_PCA import PCAStrainTraces
from pattern_recognition.utils.plot.trace_plotter import TracePlotter
from pattern_recognition.utils.utils import event_loader
from strain_analysis.align_strain_curves import AlignStrainTraces
from data.speckle_tracking.rename_ste_folders import RenameSTEFolders
from data.speckle_tracking.listFilesToRead import ListAvailableStFiles
from data.speckle_tracking.read_st_file_traces import StrainTraceFileReader
from pattern_recognition.data_preprocessing.speqle.temporalAligment import TemporalAlignment
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle
from pattern_recognition.utils.plot_dics import segment_color_dict, wall_color_dict, Cardiac_view_dict

# For speckle tracking
rename_folders = False  # Rename speckle tracking folders

# Set the following flags to True to run the corresponding code -WORKING
read_traces = False  # Save raw Strain traces from xlsx in a SQL table
interp_traces_db = False  # Align Strain Curves
concat_traces_by = 'full_traces' #'full_traces'
calculate_pca = True  # Calculate PCA

# PLOT
plot_curves = False  # Plot strain curves -WORKING
plot_interp_curves = False  # Plot strain interp -WORKING

# Currently Not Working
save_traces_in_db = False  # Save raw Strain traces from xlsx in a SQL table
data_imputation = False  # Impute missing data
plot_bad_remodeling = False  # Plot bad remodeling
plot_ecg = False  # Plot ECG

# Variables

studyType = 'BL'  # 'BL' or 'FU'
lv_remodelling = 0
speqle_traces = None
cardiac_eventDF = None
interpSelection = 'avg'  # 'max' or 'avg'
selected_source_data = 'SPEQLE'  # 'SPEQLE' OR 'ST'
Cardiac_events_order = ['IVC', 'Eject', 'IVR', 'EDF', 'ADF']


def plot_traces(table_name, isInterp, cardiac_eventT):
    global speqle_traces, trace_plotter, segment_wallDF, curve_segmentDF, cardiac_viewDF, cardiac_evDF
    if cardiac_eventT is not None:
        cardiac_eventDF = load_traces_speqle(studyType, cardiac_eventT, isImputed)
        cardiac_evDF, filtered_cardiac_ev = event_loader(
            Cardiac_events_order, cardiac_eventDF.tracesDF, interpSelection)

    speqle_traces = load_traces_speqle(studyType, table_name, isImputed)
    trace_plotter = TracePlotter(speqle_traces.tracesDF, cardiac_evDF)

    segment_wallDF = trace_plotter.group_traces('segment_wall')
    trace_plotter.plot_trace('Wall', wall_color_dict, isInterp, 1200,1400)

    curve_segmentDF = trace_plotter.group_traces('curve_segment')
    trace_plotter.plot_trace('Segment', segment_color_dict, isInterp, 2000, 1400)

    cardiac_viewDF = trace_plotter.group_traces('cardiac_view')
    trace_plotter.plot_trace('Cardiac View', Cardiac_view_dict, isInterp,1000, 1400)


if read_traces:  # read traces from different sources
    if selected_source_data == 'ST':
        if rename_folders:  # First rename speckle tracking folders
            RenameSTEFolders()
        list_st_files = ListAvailableStFiles()
        strain_traces = StrainTraceFileReader(list_st_files)
    elif selected_source_data == 'SPEQLE':
        SpeqleFileLoader(studyType)

if plot_curves:
    if selected_source_data == 'ST':
        pass
    elif selected_source_data == 'SPEQLE':
        plot_traces('speqle_segmentalStrain_T', False,None)

# Align Strain Curves and interpolate
if interp_traces_db:
    if selected_source_data == 'ST':
        aligned_traces = AlignStrainTraces()
    elif selected_source_data == 'SPEQLE':
        seg_strain_tracesDF = load_traces_speqle(studyType, 'speqle_segmentalStrain_T',isImputed)
        seg_sr_tracesDF = load_traces_speqle(studyType, 'speqle_segmentalSr_T',isImputed)
        cardiac_eventDF = load_traces_speqle(studyType, 'speqle_cardiacEvent_T',isImputed)
        strain_tracesDF = load_traces_speqle(studyType, 'speqle_strainValues_T',isImputed)

        interp_strain = TemporalAlignment(cardiac_eventDF.tracesDF, seg_strain_tracesDF.tracesDF, seg_sr_tracesDF.tracesDF,
                                          strain_tracesDF.tracesDF, interpSelection)

# Plot Interpolated Strain Curves
if plot_interp_curves:
    plot_traces('speqle_segmentalStrainInterp_T', True,'speqle_cardiacEvent_T')

# Calculate PCA
if calculate_pca:
    if selected_source_data == 'ST':
        pass
    elif selected_source_data == 'SPEQLE':
        seg_strain_interpDF = load_traces_speqle(studyType, 'speqle_segmentalStrainInterp_T',isImputed)
        tracesDF = None
        if concat_traces_by == 'full_traces':
            concatenated_strain = CurveConcatenator(seg_strain_interpDF.tracesDF, isReconstructed=False)
            tracesDF = concatenated_strain.concatenate_traces()
        elif concat_traces_by == 'segment':
            concatenated_strain = SegmentsConcatenator(seg_strain_interpDF.tracesDF)
            tracesDF = concatenated_strain.concatenate_segment_traces()

        print('Performing PCA ')
        calculate_PCA = PCAStrainTraces(tracesDF, studyType, concat_traces_by)
        if concat_traces_by == 'full_traces':
            calculate_PCA.prepare_all_traces()
            pca_result = calculate_PCA.calculate_pca()
            calculate_PCA.reconstruct_traces(None)
            pass

        elif concat_traces_by == 'segment':
            calculate_PCA.prepare_segment_traces()
            pass