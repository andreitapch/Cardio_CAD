import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.offline as pyo
import plotly.io as py
from scipy import interpolate
import plotly.graph_objects as go
from pattern_recognition.data.insert_to_db.insertSpeqleDB import InsertInterpTracesToDB
from pattern_recognition.utils.parameters_dics import parameters
from pattern_recognition.utils.utils import event_loader


def interpolate_traces(df, cardiac_ev_interp, cardiac_events, events_order):
    interpDF = df.copy()
    interpDF[['strain_trace', 'curve_length']] = None
    unique_study_nr = df['study_nr'].unique()

    for study_nr in tqdm(unique_study_nr):
        studyDf = df[df['study_nr'] == study_nr]
        df.loc[df['study_nr'] == study_nr, 'segment_wall'] = df.loc[
            df['study_nr'] == study_nr, 'segment_wall'].str.strip()

        unique_Segment_wall = studyDf['segment_wall'].str.strip().unique()
        for idx_seg, segment_wall in enumerate(unique_Segment_wall):

            split_traces = []
            studyDf.loc[:, 'segment_wall'] = studyDf['segment_wall'].str.strip()
            study_seg_wallDF = studyDf[studyDf['segment_wall'] == segment_wall]

            cardiac_ev_studyDF = cardiac_events[(cardiac_events['study_nr'] == study_nr) &
                                                (cardiac_events['segment_wall'] == segment_wall)]
            if not cardiac_ev_studyDF['event_time'].is_monotonic_increasing:
                continue

            if cardiac_ev_studyDF['event_time'].duplicated().any():
                print(f"Study {study_nr} segment {segment_wall} has the same timming")
                continue

            traces = study_seg_wallDF['strain_trace'].apply(lambda t_x: [float(i) for i in ast.literal_eval(t_x)])
            idx_row_df = traces.index
            interpolated_trace = {event: [] for event in cardiac_ev_interp['event_name'].unique()}
            prev_event_time = None
            concatenated_trace = []
            for idx_event, event in enumerate(cardiac_ev_studyDF['event_name']):

                ev_length_in = cardiac_ev_interp.loc[cardiac_ev_interp['event_name'] == event, 'event_length'].values[0]
                ev_time_orig = cardiac_ev_studyDF.loc[cardiac_ev_studyDF['event_name'] == event, 'event_time'].values[0]

                if event == 'IVC':
                    ev_length_in = cardiac_ev_interp.loc[cardiac_ev_interp[
                                                             'event_name'] == event, 'event_time'].values[0]
                    split_traces = [trace[:ev_time_orig] for trace in traces]
                elif event == 'Eject':
                    split_traces = [trace[prev_event_time:ev_time_orig] for trace in traces]
                elif event == 'IVR':
                    split_traces = [trace[prev_event_time:ev_time_orig] for trace in traces]
                elif event == 'EDF':
                    split_traces = [trace[prev_event_time:ev_time_orig] for trace in traces]
                elif event == 'ADF':
                    split_traces = [trace[prev_event_time:ev_time_orig] for trace in traces]
                prev_event_time = ev_time_orig

                for trace in split_traces:
                    x = np.arange(len(trace))
                    f = interpolate.interp1d(x, trace)
                    x_new = np.linspace(0, len(trace) - 1, num=ev_length_in - 1, endpoint=True)
                    interp_trace = f(x_new)
                    interpolated_trace[event].append(interp_trace)

            for idx_trace, idx_df in zip(range(len(interpolated_trace['IVC'])), idx_row_df):
                interp_traces = []
                for event in events_order:
                    interp_traces.extend(np.round(interpolated_trace[event][idx_trace], 6))
                interpDF.loc[idx_df, 'strain_trace'] = str(interp_traces)
                interpDF.loc[idx_df, 'curve_length'] = int(len(interp_traces))
    interpDF['isImputed'] = 0
    return interpDF


def plot_trace(cardiac_ev_interp, concatenated_trace, traces):
    dir_path = parameters['save_plot_dir']
    cardiac_ev_interp['event_time'].values[-1] = len(concatenated_trace) - 1
    fig = go.Figure()

    for idx, (x, label) in enumerate(
            zip(cardiac_ev_interp['event_time'].values, cardiac_ev_interp['event_name'].values)):
        fig.add_shape(type="line", x0=x, x1=x, yref="paper", y0=0, y1=1, line=dict(color="red", dash="dash"))
        fig.add_annotation(x=x, y=1.1, text=label, showarrow=False, yref="paper",
                           font=dict(size=10, color="black", family="Arial, bold"))

    fig.add_trace(go.Scatter(y=traces[0], mode='lines', name='Original', line=dict(color='black')))
    fig.add_trace(go.Scatter(y=concatenated_trace, mode='lines', name='Interpolated', line=dict(color='blue')))
    py.write_image(fig, os.path.join(dir_path, 'InterpVSOriginal.png'))


def calculate_gls(strain_tracesDF, segmental_tracesDF):
    interp_glsDF = strain_tracesDF.copy()
    unique_study_nr = segmental_tracesDF['study_nr'].unique()
    for study_nr in tqdm(unique_study_nr):
        studyDf = segmental_tracesDF[segmental_tracesDF['study_nr'] == study_nr]
        studyDf = studyDf[studyDf['strain_trace'].notna()]
        studyDf = studyDf[studyDf['strain_trace'].apply(lambda x: isinstance(x, str))]
        try:
            new_df = pd.DataFrame(studyDf['strain_trace'].apply(ast.literal_eval).tolist())
        except ValueError:
            print(f"Error parsing 'strain_trace' for study_nr {study_nr}")
            continue
        new_df = pd.DataFrame(studyDf['strain_trace'].apply(ast.literal_eval).tolist())
        column_sums = new_df.sum()
        num_rows = len(new_df)
        strain_trace = column_sums / num_rows
        interp_glsDF.loc[interp_glsDF['study_nr'] == study_nr, 'gls'] = str(strain_trace.tolist())
    interp_glsDF['isImputed'] = 0
    return interp_glsDF


class TemporalAlignment:
    def __init__(self, cardiac_eventDF, seg_strain_tracesDF, seg_sr_tracesDF, strain_tracesDF, interpSelection):

        self.grouped_ev = pd.DataFrame()
        self.cardiac_evDF = pd.DataFrame()
        self.filtered_cardiac_ev = pd.DataFrame()
        self.Cardiac_events_order = ['IVC', 'Eject', 'IVR', 'EDF', 'ADF']  # 'EMC',
        self.cardiac_evDF, self.filtered_cardiac_ev = event_loader(
            self.Cardiac_events_order, cardiac_eventDF, interpSelection)

        # print('Interpolating segmental strain traces....')
        segmental_st_interp = interpolate_traces(seg_strain_tracesDF, self.cardiac_evDF, self.filtered_cardiac_ev,
                                                 self.Cardiac_events_order)
        print('Inserting into db Interpolated segmental strain traces....')
        InsertInterpTracesToDB(segmental_st_interp, 'speqle_segmentalStrainInterp_T', True)

        print('Interpolating segmental strain rate traces....')
        segmental_sr_interp = interpolate_traces(seg_sr_tracesDF, self.cardiac_evDF, self.filtered_cardiac_ev,
                                                 self.Cardiac_events_order)
        print('Inserting into db Interpolated segmental strain traces....')
        InsertInterpTracesToDB(segmental_sr_interp, 'speqle_segmentalSrInterp_T', True)

        print('Interpolating GLS traces....')
        gls_interp = calculate_gls(strain_tracesDF, segmental_st_interp)

        print('Inserting into db Interpolating segmental strain traces....')
        InsertInterpTracesToDB(gls_interp, 'speqle_strainValuesInterp_T', False)


